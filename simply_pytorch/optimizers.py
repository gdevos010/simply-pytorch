"""PyTorch optimizers with Cautious Weight Decay support.

This module provides PyTorch implementations of optimizers from the Simply
codebase, including SGD, Adam, Lion, and Muon. All optimizers support both
standard weight decay and Cautious Weight Decay (CWD) as described in
https://arxiv.org/html/2510.12402v1

Reference JAX implementation: https://github.com/google-deepmind/simply/blob/main/simply/utils/optimizers.py
"""

from collections.abc import Callable

import torch

from torch.optim.optimizer import Optimizer

from simply_pytorch.utils import newton_schulz_orthogonalize

MIN_PARAM_NDIM = 2


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Simple gradient descent without momentum, following the JAX Simply implementation.
    Supports both standard and cautious weight decay.

    Reference: JAX implementation lines 92-102

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        weight_decay: Weight decay coefficient (L2 penalty) (default: 0.0)
        use_cautious_wd: Whether to use Cautious Weight Decay (default: False)

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Callable | list,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_cautious_wd: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "use_cautious_wd": use_cautious_wd,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if group["weight_decay"] != 0:
                    if group["use_cautious_wd"]:
                        # Cautious Weight Decay: only apply where signs align
                        mask = grad * p.data >= 0
                        p.data[mask] -= (
                            group["lr"] * group["weight_decay"] * p.data[mask]
                        )
                    else:
                        # Standard weight decay
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.add_(grad, alpha=-group["lr"])

        return loss


class Adam(Optimizer):
    """Adam optimizer with bias correction.

    Implements the Adam algorithm as described in the original paper with bias correction.
    Supports both standard and cautious weight decay.

    Reference: JAX implementation lines 107-138, CWD Paper Appendix D.4
    Paper: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015)

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-6)
        weight_decay: Weight decay coefficient (L2 penalty) (default: 0.0)
        use_cautious_wd: Whether to use Cautious Weight Decay (default: False)

    Example:
        >>> optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Callable | list,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        use_cautious_wd: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_cautious_wd": use_cautious_wd,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization (lazy)
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute step
                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = bias_correction2**0.5

                # Compute update direction
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
                update = exp_avg / denom

                # Apply weight decay
                if group["weight_decay"] != 0:
                    if group["use_cautious_wd"]:
                        # Cautious Weight Decay: only apply where signs align
                        mask = update * p.data >= 0
                        p.data[mask] -= (
                            group["lr"] * group["weight_decay"] * p.data[mask]
                        )
                    else:
                        # Standard weight decay
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.add_(update, alpha=-step_size)

        return loss


class Lion(Optimizer):
    """Lion optimizer (EvoLved Sign Momentum).

    Lion is a simple yet effective optimizer that uses sign-based updates with
    momentum. It's memory-efficient and often outperforms AdamW.

    Reference: JAX implementation lines 143-169, CWD Paper Appendix B.3
    Paper: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-4, typically 3-10x smaller than AdamW)
        betas: Coefficients for computing running averages (default: (0.95, 0.98))
        weight_decay: Weight decay coefficient (default: 0.0)
        use_cautious_wd: Whether to use Cautious Weight Decay (default: False)

    Example:
        >>> optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

    Note:
        Lion typically requires a learning rate 3-10x smaller than AdamW but can
        use a higher weight decay coefficient for better regularization.
    """

    def __init__(
        self,
        params: Callable | list,
        lr: float = 1e-4,
        betas: tuple = (0.95, 0.98),
        weight_decay: float = 0.0,
        use_cautious_wd: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "use_cautious_wd": use_cautious_wd,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization (lazy)
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]

                # Compute intermediate momentum m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                # This is used for both the sign update and the subsequent beta2 decay
                m_t = exp_avg * beta1 + grad * (1 - beta1)

                # Compute update as sign of m_t
                update = torch.sign(m_t)

                # Update momentum buffer with beta2 decay applied to m_t
                # m_{t} ← beta2 * m_{t-1} + (1 - beta2) * m_t
                exp_avg.mul_(beta2).add_(m_t, alpha=1 - beta2)

                # Apply weight decay
                if group["weight_decay"] != 0:
                    if group["use_cautious_wd"]:
                        # Cautious Weight Decay: only apply where signs align
                        mask = update * p.data >= 0
                        p.data[mask] -= (
                            group["lr"] * group["weight_decay"] * p.data[mask]
                        )
                    else:
                        # Standard weight decay
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.add_(update, alpha=-group["lr"])

        return loss


class AdamAtan2(Optimizer):
    """Adam-atan2 optimizer with scale invariance.

    Adam-atan2 replaces the traditional Adam update rule with atan2 to achieve
    scale invariance and remove the need for epsilon hyperparameter. This leads
    to improved numerical stability across different parameter scales.

    Reference: https://github.com/lucidrains/adam-atan2-pytorch
    Paper: "Scaling Exponents Across Parameterizations and Optimizers" (Everett et al., 2024)

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-4)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        weight_decay: Weight decay coefficient (L2 penalty) (default: 0.0)
        use_cautious_wd: Whether to use Cautious Weight Decay (default: False)

    Example:
        >>> optimizer = AdamAtan2(model.parameters(), lr=1e-4, weight_decay=1e-3)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

    Note:
        Unlike traditional Adam, this optimizer doesn't require an epsilon parameter
        for numerical stability, as the atan2 function naturally handles the scale.
    """

    def __init__(
        self,
        params: Callable | list,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        use_cautious_wd: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "use_cautious_wd": use_cautious_wd,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization (lazy)
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Bias-corrected estimates
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2

                # Compute update using atan2 for scale invariance
                # atan2(y, x) computes arctan(y/x) with correct quadrant handling
                update = torch.atan2(exp_avg_corrected, exp_avg_sq_corrected.sqrt())

                # Apply weight decay
                if group["weight_decay"] != 0:
                    if group["use_cautious_wd"]:
                        # Cautious Weight Decay: only apply where signs align
                        mask = update * p.data >= 0
                        p.data[mask] -= (
                            group["lr"] * group["weight_decay"] * p.data[mask]
                        )
                    else:
                        # Standard weight decay
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.add_(update, alpha=-group["lr"])

        return loss


class Muon(Optimizer):
    """Muon optimizer (Momentum Orthogonalized by Newton-schulz).

    Muon is a hybrid optimizer that uses Newton-Schulz orthogonalization for
    weight matrices (ndim >= 2, max_dim <= dim_threshold) and Adam for other
    parameters (biases, layer norms, embeddings with large dimensions).

    Reference: JAX implementation lines 175-389
    Paper: "Muon is Scalable for LLM Training" https://arxiv.org/html/2502.16982v1

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.02 for LLM pre-training)
        momentum: Momentum coefficient for Muon updates (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        adam_betas: Betas for Adam fallback (default: (0.9, 0.95))
        adam_eps: Epsilon for Adam fallback (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
        use_cautious_wd: Whether to use Cautious Weight Decay (default: False)
        dim_threshold: Maximum dimension for using Muon (default: 10000)

    Example:
        >>> optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.01)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

    Note:
        Muon works best for transformer models where weight matrices benefit from
        orthogonalization while keeping Adam for other parameters.
    """

    def __init__(
        self,
        params: Callable | list,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_betas: tuple = (0.9, 0.95),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_cautious_wd: bool = False,
        dim_threshold: int = 10000,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= adam_betas[0] < 1.0:
            raise ValueError(f"Invalid Adam beta parameter at index 0: {adam_betas[0]}")
        if not 0.0 <= adam_betas[1] < 1.0:
            raise ValueError(f"Invalid Adam beta parameter at index 1: {adam_betas[1]}")
        if adam_eps < 0.0:
            raise ValueError(f"Invalid Adam epsilon value: {adam_eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adam_betas": adam_betas,
            "adam_eps": adam_eps,
            "weight_decay": weight_decay,
            "use_cautious_wd": use_cautious_wd,
            "dim_threshold": dim_threshold,
        }
        super().__init__(params, defaults)

    def _compute_muon_update(
        self,
        grad: torch.Tensor,
        state: dict,
        group: dict,
    ) -> torch.Tensor:
        """Compute Muon optimizer update using Newton-Schulz orthogonalization."""
        momentum_buffer = state["momentum_buffer"]
        momentum_buffer.mul_(group["momentum"]).add_(grad)

        if group["nesterov"]:
            update_input = momentum_buffer * group["momentum"] + grad
        else:
            update_input = momentum_buffer

        return newton_schulz_orthogonalize(
            update_input,
            num_steps=group["ns_steps"],
            eps=group["adam_eps"],
        )

    def _compute_adam_update(
        self,
        grad: torch.Tensor,
        state: dict,
        adam_beta1: float,
        adam_beta2: float,
        group: dict,
    ) -> torch.Tensor:
        """Compute Adam optimizer update."""
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

        exp_avg.mul_(adam_beta1).add_(grad, alpha=1 - adam_beta1)
        exp_avg_sq.mul_(adam_beta2).addcmul_(grad, grad, value=1 - adam_beta2)

        bias_correction1 = 1 - adam_beta1 ** state["step"]
        bias_correction2 = 1 - adam_beta2 ** state["step"]

        step_size = 1.0 / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["adam_eps"])
        return step_size * exp_avg / denom

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            adam_beta1, adam_beta2 = group["adam_betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Determine if this parameter should use Muon or Adam
                use_muon = (
                    p.ndim >= MIN_PARAM_NDIM and max(p.shape) <= group["dim_threshold"]
                )

                # State initialization (lazy)
                if len(state) == 0:
                    state["step"] = 0
                    state["use_muon"] = use_muon

                    if use_muon:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
                    else:
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Compute update based on optimizer type
                if state["use_muon"]:
                    update = self._compute_muon_update(grad, state, group)
                else:
                    update = self._compute_adam_update(
                        grad, state, adam_beta1, adam_beta2, group
                    )

                # Apply weight decay
                if group["weight_decay"] != 0:
                    if group["use_cautious_wd"]:
                        mask = update * p.data >= 0
                        p.data[mask] -= (
                            group["lr"] * group["weight_decay"] * p.data[mask]
                        )
                    else:
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.add_(update, alpha=-group["lr"])

        return loss
