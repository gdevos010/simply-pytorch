"""Tests for Simply PyTorch optimizers.

Port of the JAX Simply optimizers_test.py to PyTorch, with additional tests
for Cautious Weight Decay functionality.

Reference: /workspace/condor/other/simply/simply/utils/optimizers_test.py
"""

import pytest
import torch

from simply_pytorch import SGD, Adam, Lion, Muon

# Constants for testing
GRADIENT_TEST_VALUE = 2.0
PARAM_POSITIVE_VALUE = 2.0
PARAM_NEGATIVE_VALUE = -2.0


class TestOptimizers:
    """Test suite for Simply PyTorch optimizers."""

    def test_sgd_basic(self):
        """Test basic SGD functionality."""
        # Create a simple parameter
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = SGD([param], lr=1.0)

        # Initial state
        assert param.item() == 1.0

        # Compute gradient
        loss = param * 2.0
        loss.backward()

        # Check gradient
        assert param.grad.item() == GRADIENT_TEST_VALUE

        # Take optimization step
        optimizer.step()

        # Parameter should be updated: 1.0 - 1.0 * 2.0 = -1.0
        torch.testing.assert_close(param.item(), -1.0, rtol=1e-5, atol=1e-5)

    def test_adam_basic(self):
        """Test basic Adam functionality with bias correction."""
        # Create a simple parameter
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = Adam([param], lr=1.0, betas=(0.9, 0.999), eps=1e-6)

        # Initial state
        assert param.item() == 1.0

        # Compute gradient
        loss = param * 2.0
        loss.backward()

        # Take optimization step
        optimizer.step()

        # Check state was created
        assert len(optimizer.state[param]) > 0
        assert "exp_avg" in optimizer.state[param]
        assert "exp_avg_sq" in optimizer.state[param]
        assert optimizer.state[param]["step"] == 1

        # Check momentum values
        # m = 0.9 * 0 + 0.1 * 2.0 = 0.2
        torch.testing.assert_close(
            optimizer.state[param]["exp_avg"].item(), 0.2, rtol=1e-5, atol=1e-5
        )
        # v = 0.999 * 0 + 0.001 * 4.0 = 0.004
        torch.testing.assert_close(
            optimizer.state[param]["exp_avg_sq"].item(), 0.004, rtol=1e-5, atol=1e-5
        )

        # Expected update with bias correction:
        # m_hat = 0.2 / (1 - 0.9^1) = 0.2 / 0.1 = 2.0
        # v_hat = 0.004 / (1 - 0.999^1) = 0.004 / 0.001 = 4.0
        # update = m_hat / (sqrt(v_hat) + eps) = 2.0 / (2.0 + 1e-6) ≈ 1.0
        # new_param = 1.0 - 1.0 * 1.0 = 0.0
        torch.testing.assert_close(param.item(), 0.0, rtol=1e-4, atol=1e-4)

    def test_lion_basic(self):
        """Test basic Lion functionality."""
        # Create a simple parameter
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = Lion([param], lr=1.0, betas=(0.95, 0.98))

        # Initial state
        assert param.item() == 1.0

        # Compute gradient
        loss = param * 2.0
        loss.backward()

        # Take optimization step
        optimizer.step()

        # Check state was created
        assert len(optimizer.state[param]) > 0
        assert "exp_avg" in optimizer.state[param]

        # Check momentum value
        # Lion uses two-stage momentum:
        # m_t = beta1 * 0 + (1 - beta1) * 2.0 = 0.95 * 0 + 0.05 * 2.0 = 0.1
        # exp_avg = beta2 * 0 + (1 - beta2) * m_t = 0.98 * 0 + 0.02 * 0.1 = 0.002
        torch.testing.assert_close(
            optimizer.state[param]["exp_avg"].item(), 0.002, rtol=1e-4, atol=1e-4
        )

        # Expected update:
        # update = sign(0.95 * 0 + 0.05 * 2.0) = sign(0.1) = 1.0
        # new_param = 1.0 - 1.0 * 1.0 = 0.0
        torch.testing.assert_close(param.item(), 0.0, rtol=1e-5, atol=1e-5)

    def test_muon_with_2d_param(self):
        """Test Muon with 2D parameter (should use Newton-Schulz)."""
        # Create a 2D weight matrix
        param = torch.randn(4, 4, requires_grad=True)
        optimizer = Muon([param], lr=0.01, dim_threshold=10000)

        initial_param = param.data.clone()

        # Compute gradient
        loss = param.sum()
        loss.backward()

        # Take optimization step
        optimizer.step()

        # Check state was created for Muon
        assert len(optimizer.state[param]) > 0
        assert optimizer.state[param]["use_muon"] is True
        assert "momentum_buffer" in optimizer.state[param]

        # Parameter should have changed
        assert not torch.allclose(param.data, initial_param)

    def test_muon_with_1d_param(self):
        """Test Muon with 1D parameter (should use Adam fallback)."""
        # Create a 1D bias vector
        param = torch.randn(4, requires_grad=True)
        optimizer = Muon([param], lr=0.01, dim_threshold=10000)

        initial_param = param.data.clone()

        # Compute gradient
        loss = param.sum()
        loss.backward()

        # Take optimization step
        optimizer.step()

        # Check state was created for Adam
        assert len(optimizer.state[param]) > 0
        assert optimizer.state[param]["use_muon"] is False
        assert "exp_avg" in optimizer.state[param]
        assert "exp_avg_sq" in optimizer.state[param]

        # Parameter should have changed
        assert not torch.allclose(param.data, initial_param)

    def test_cautious_weight_decay_sgd(self):
        """Test Cautious Weight Decay with SGD."""
        # Create parameters with specific values
        param = torch.tensor([1.0, -1.0, 2.0, -2.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1, weight_decay=0.5, use_cautious_wd=True)

        # Set gradient: [1.0, 1.0, -1.0, -1.0]
        # Signs align for indices 0 and 3, not for 1 and 2
        param.grad = torch.tensor([1.0, 1.0, -1.0, -1.0])

        param.data.clone()

        # Take step
        optimizer.step()

        # With CWD:
        # Index 0: update=1.0, param=1.0, signs match -> apply decay
        #   param = 1.0 - 0.1*0.5*1.0 - 0.1*1.0 = 1.0 - 0.05 - 0.1 = 0.85
        # Index 1: update=1.0, param=-1.0, signs differ -> no decay
        #   param = -1.0 - 0.1*1.0 = -1.1
        # Index 2: update=-1.0, param=2.0, signs differ -> no decay
        #   param = 2.0 - 0.1*(-1.0) = 2.1
        # Index 3: update=-1.0, param=-2.0, signs match -> apply decay
        #   param = -2.0 - 0.1*0.5*(-2.0) - 0.1*(-1.0) = -2.0 + 0.1 + 0.1 = -1.8

        expected = torch.tensor([0.85, -1.1, 2.1, -1.8])
        torch.testing.assert_close(param.data, expected, rtol=1e-5, atol=1e-5)

    def test_standard_weight_decay_sgd(self):
        """Test standard weight decay with SGD."""
        # Create parameters with specific values
        param = torch.tensor([1.0, -1.0, 2.0, -2.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1, weight_decay=0.5, use_cautious_wd=False)

        # Set gradient
        param.grad = torch.tensor([1.0, 1.0, -1.0, -1.0])

        # Take step
        optimizer.step()

        # With standard WD, decay applied to all parameters:
        # param = param - lr*wd*param - lr*grad
        # Index 0: 1.0 - 0.1*0.5*1.0 - 0.1*1.0 = 0.85
        # Index 1: -1.0 - 0.1*0.5*(-1.0) - 0.1*1.0 = -1.0 + 0.05 - 0.1 = -1.05
        # Index 2: 2.0 - 0.1*0.5*2.0 - 0.1*(-1.0) = 2.0 - 0.1 + 0.1 = 2.0
        # Index 3: -2.0 - 0.1*0.5*(-2.0) - 0.1*(-1.0) = -2.0 + 0.1 + 0.1 = -1.8

        expected = torch.tensor([0.85, -1.05, 2.0, -1.8])
        torch.testing.assert_close(param.data, expected, rtol=1e-5, atol=1e-5)

    def test_cautious_weight_decay_adam(self):
        """Test Cautious Weight Decay with Adam."""
        param = torch.tensor([2.0, -2.0], requires_grad=True)
        initial_param = param.data.clone()
        optimizer = Adam([param], lr=0.1, weight_decay=0.1, use_cautious_wd=True)

        # Positive gradient (same sign as param[0], opposite to param[1])
        param.grad = torch.tensor([1.0, 1.0])

        optimizer.step()

        # CWD should only apply decay to param[0] where signs align
        # param[0] should change more from initial value due to weight decay
        change_0 = abs(param[0].item() - initial_param[0].item())
        change_1 = abs(param[1].item() - initial_param[1].item())
        assert change_0 > change_1, (
            f"param[0] change: {change_0}, param[1] change: {change_1}"
        )

    def test_cautious_weight_decay_lion(self):
        """Test Cautious Weight Decay with Lion."""
        param = torch.tensor([2.0, -2.0], requires_grad=True)
        optimizer = Lion([param], lr=0.1, weight_decay=0.1, use_cautious_wd=True)

        # Positive gradient (same sign as param[0], opposite to param[1])
        param.grad = torch.tensor([1.0, 1.0])

        optimizer.step()

        # CWD should only apply decay to param[0] where signs align
        # Both will decrease, but param[0] should decrease more due to weight decay
        assert param[0].item() < PARAM_POSITIVE_VALUE
        assert param[1].item() < PARAM_NEGATIVE_VALUE

    def test_optimizer_state_dict_adam(self):
        """Test state_dict save/load for Adam."""
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = Adam([param], lr=0.1)

        # Take a step to create state
        param.grad = torch.tensor([1.0])
        optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer and load state
        param2 = torch.tensor([1.0], requires_grad=True)
        optimizer2 = Adam([param2], lr=0.1)
        optimizer2.load_state_dict(state_dict)

        # States should match
        assert optimizer2.state[param2]["step"] == optimizer.state[param]["step"]
        torch.testing.assert_close(
            optimizer2.state[param2]["exp_avg"], optimizer.state[param]["exp_avg"]
        )

    def test_param_groups(self):
        """Test multiple parameter groups with different learning rates."""
        param1 = torch.tensor([1.0], requires_grad=True)
        param2 = torch.tensor([1.0], requires_grad=True)

        optimizer = Adam(
            [
                {"params": [param1], "lr": 0.1},
                {"params": [param2], "lr": 0.01},
            ]
        )

        # Set gradients
        param1.grad = torch.tensor([1.0])
        param2.grad = torch.tensor([1.0])

        # Take step
        optimizer.step()

        # param1 should change more than param2 due to higher learning rate
        assert abs(param1.item() - 1.0) > abs(param2.item() - 1.0)

    def test_sgd_with_zero_grad(self):
        """Test that optimizer handles zero gradients correctly."""
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1)

        param.grad = torch.tensor([0.0])
        optimizer.step()

        # Parameter should not change with zero gradient and no weight decay
        torch.testing.assert_close(param.item(), 1.0, rtol=1e-5, atol=1e-5)

    def test_adam_convergence(self):
        """Test that Adam converges on a simple quadratic."""
        # Minimize f(x) = (x - 3)^2
        param = torch.tensor([0.0], requires_grad=True)
        optimizer = Adam([param], lr=0.1)

        for _ in range(100):
            optimizer.zero_grad()
            loss = (param - 3.0) ** 2
            loss.backward()
            optimizer.step()

        # Should converge close to 3.0
        torch.testing.assert_close(param.item(), 3.0, rtol=1e-2, atol=1e-2)

    def test_lion_sign_behavior(self):
        """Test that Lion produces sign-based updates."""
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = Lion([param], lr=1.0)

        # Large positive gradient
        param.grad = torch.tensor([100.0])
        optimizer.step()
        update1 = 1.0 - param.item()

        # Reset
        param.data = torch.tensor([1.0])
        optimizer = Lion([param], lr=1.0)

        # Small positive gradient
        param.grad = torch.tensor([0.01])
        optimizer.step()
        update2 = 1.0 - param.item()

        # Both should produce same update magnitude due to sign operation
        torch.testing.assert_close(update1, update2, rtol=1e-5, atol=1e-5)

    def test_muon_dim_threshold(self):
        """Test that Muon respects dim_threshold parameter."""
        # Create parameter with max dim > threshold
        param = torch.randn(100, 100, requires_grad=True)
        optimizer = Muon([param], lr=0.01, dim_threshold=50)

        param.grad = torch.randn_like(param)
        optimizer.step()

        # Should use Adam because max(100, 100) > 50
        assert optimizer.state[param]["use_muon"] is False

        # Create parameter with max dim <= threshold
        param2 = torch.randn(10, 10, requires_grad=True)
        optimizer2 = Muon([param2], lr=0.01, dim_threshold=50)

        param2.grad = torch.randn_like(param2)
        optimizer2.step()

        # Should use Muon because max(10, 10) <= 50
        assert optimizer2.state[param2]["use_muon"] is True

    def test_invalid_learning_rate(self):
        """Test that invalid learning rates raise errors."""
        param = torch.tensor([1.0], requires_grad=True)

        with pytest.raises(ValueError, match="Invalid"):
            SGD([param], lr=-0.1)

        with pytest.raises(ValueError, match="Invalid"):
            Adam([param], lr=-0.1)

    def test_invalid_betas(self):
        """Test that invalid beta values raise errors."""
        param = torch.tensor([1.0], requires_grad=True)

        with pytest.raises(ValueError, match="Invalid"):
            Adam([param], lr=0.1, betas=(1.5, 0.999))

        with pytest.raises(ValueError, match="Invalid"):
            Lion([param], lr=0.1, betas=(0.95, -0.1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
