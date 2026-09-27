"""Actual MLX safetensors interchange and short numerical cross-backend checks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten
from nanochat_mlx.hybrid.model import HybridLM as MLXModel
from nanochat_mlx.hybrid.checkpoint import save_checkpoint as save_mlx
from nanochat_cuda.model import HybridLM as TorchModel
from nanochat_cuda.checkpoint import initialize_weights
from tests.cuda.test_model import tiny_config


def test_actual_mlx_weights_logits_and_gradients(tmp_path):
    torch.set_num_threads(1)
    config = tiny_config()
    mlx_model = MLXModel(config)
    path = save_mlx(tmp_path, mlx_model, 0, {"fixture": True})
    model = TorchModel(config, "reference")
    initialize_weights(model, path, {"fixture": True})
    ids = np.random.default_rng(42).integers(0, 97, (1, 9), dtype=np.int32)
    targets = np.roll(ids, -1, axis=1).copy()
    targets[:, :2] = -1
    expected = mlx_model(mx.array(ids))
    actual = model(torch.tensor(ids, dtype=torch.long))
    np.testing.assert_allclose(
        actual.detach().numpy(), np.array(expected), atol=3e-6, rtol=4e-5
    )
    loss, gradients = nn.value_and_grad(
        mlx_model, lambda m: m(mx.array(ids), targets=mx.array(targets))
    )(mlx_model)
    actual_loss = model(
        torch.tensor(ids, dtype=torch.long), targets=torch.tensor(targets)
    )
    actual_loss.backward()
    np.testing.assert_allclose(actual_loss.item(), loss.item(), atol=2e-6)
    params = dict(model.named_parameters())
    for name, grad in tree_flatten(gradients):
        np.testing.assert_allclose(
            params[name].grad.numpy(),
            np.array(grad),
            atol=5e-6,
            rtol=3e-3,
            err_msg=name,
        )
