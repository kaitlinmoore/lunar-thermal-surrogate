"""GPU smoke test: proves PyTorch and PhysicsNeMo work on this machine's GPU.

This checks the environment, not project logic. It confirms that the installed
PyTorch build carries kernels for the RTX 5080 (Blackwell, compute capability
12.0, "sm_120"), and that PhysicsNeMo's 1D Fourier neural operator, the model
family the surrogate uses, can run a forward and a backward pass on CUDA.
Skipped when CUDA is unavailable, so `uv run pytest` still passes on a CPU-only
machine.
"""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


def test_torch_build_targets_blackwell():
    """A build without sm_120 would fail with "no kernel image" at the first op."""
    assert "sm_120" in torch.cuda.get_arch_list()


def test_fno_1d_forward_and_backward_on_cuda():
    """Build a small 1D FNO, push a batch through it, and backpropagate.

    Shapes mirror the planned surrogate: a few input channels (absorbed flux
    plus broadcast scalar parameters) over a time axis, one output channel
    (surface temperature) on the same axis.
    """
    from physicsnemo.models.fno import FNO

    torch.manual_seed(0)
    device = torch.device("cuda")
    in_channels, out_channels, n_time, batch = 3, 1, 256, 4

    model = FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        dimension=1,
        latent_channels=16,
        num_fno_layers=2,
        num_fno_modes=8,
        decoder_layers=1,
        decoder_layer_size=16,
    ).to(device)

    x = torch.randn(batch, in_channels, n_time, device=device)
    y = model(x)
    assert y.shape == (batch, out_channels, n_time)
    assert y.device.type == "cuda"

    loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
    loss.backward()
    assert torch.isfinite(loss)

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
