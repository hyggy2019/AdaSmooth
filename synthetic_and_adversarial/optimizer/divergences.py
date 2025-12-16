"""
Divergence functionals for AdaSmoothES.

This module implements various divergence measures that can be used
to regularize the policy optimization objective:

    min E_π[F(x)] + (1/β) * D(π || π_ref)

Different divergences lead to different weight computations.

Author: Based on theoretical framework in KL-regularized optimization
"""

import torch
import math
from abc import ABC, abstractmethod


class Divergence(ABC):
    """Base class for divergence functionals."""

    @abstractmethod
    def compute_weights(self, f_values: torch.Tensor, beta: float, baseline: float) -> torch.Tensor:
        """
        Compute importance weights from function values.

        Args:
            f_values: Function values of shape (K,)
            beta: Temperature parameter (inverse regularization strength)
            baseline: Baseline value for variance reduction

        Returns:
            weights: Normalized weights summing to 1
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of the divergence."""
        pass


class KLDivergence(Divergence):
    """
    Forward KL divergence: KL(π || π_ref).

    Optimal policy: π*(x) ∝ π_ref(x) · exp(-β·F(x))
    Weights: w_k = softmax(-β·(f_k - b))

    Properties:
    - Mode-seeking: prefers to cover modes of low function values
    - Boltzmann distribution
    - Most commonly used in RL and evolution strategies
    """

    def compute_weights(self, f_values: torch.Tensor, beta: float, baseline: float) -> torch.Tensor:
        advantages = f_values - baseline
        log_weights = -advantages / beta  # FIX: was -beta * advantages (WRONG!)

        # Log-sum-exp trick for numerical stability
        max_log_w = log_weights.max()
        log_weights_stable = log_weights - max_log_w
        weights = torch.exp(log_weights_stable)
        weights = weights / weights.sum()

        # Handle numerical issues
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            weights = torch.ones_like(weights) / len(weights)

        return weights

    def name(self) -> str:
        return "KL"


class ReverseKLDivergence(Divergence):
    """
    Reverse KL divergence: KL(π_ref || π).

    Optimal policy: π*(x) ∝ π_ref(x) / (1 + β·F(x))
    Weights: w_k ∝ 1 / (1 + β·(f_k - b))

    Properties:
    - Mean-seeking: prefers to match mean of reference distribution
    - More robust to outliers than forward KL
    - Used in variational inference
    """

    def compute_weights(self, f_values: torch.Tensor, beta: float, baseline: float) -> torch.Tensor:
        advantages = f_values - baseline

        # Compute weights: w_k ∝ 1 / (1 + A_k/β)
        # FIX: was (1 + beta * advantages), now (1 + advantages / beta)
        weights = 1.0 / (1.0 + advantages / beta + 1e-8)
        weights = torch.clamp(weights, min=0.0)  # Ensure non-negative

        # Normalize
        weights_sum = weights.sum()
        if weights_sum > 1e-8:
            weights = weights / weights_sum
        else:
            weights = torch.ones_like(weights) / len(weights)

        return weights

    def name(self) -> str:
        return "ReverseKL"


class ChiSquaredDivergence(Divergence):
    """
    χ² divergence: (1/2) * E[(π/π_ref - 1)²].

    Optimal policy: π*(x) ∝ π_ref(x) · (1 - β·F(x))
    Weights: w_k ∝ max(0, 1 - β·(f_k - b))

    Properties:
    - Quadratic penalty
    - More robust than KL for large deviations
    - Allows negative advantages naturally
    """

    def compute_weights(self, f_values: torch.Tensor, beta: float, baseline: float) -> torch.Tensor:
        advantages = f_values - baseline

        # Compute weights: w_k ∝ max(0, 1 - A_k/β)
        # FIX: was (1 - beta * advantages), now (1 - advantages / beta)
        weights = torch.clamp(1.0 - advantages / beta, min=0.0)

        # Normalize
        weights_sum = weights.sum()
        if weights_sum > 1e-8:
            weights = weights / weights_sum
        else:
            weights = torch.ones_like(weights) / len(weights)

        return weights

    def name(self) -> str:
        return "ChiSquared"


class RenyiDivergence(Divergence):
    """
    Rényi divergence: D_α(π || π_ref) = (1/(α-1)) log E[(π/π_ref)^α].

    Optimal policy: π*(x) ∝ π_ref(x) · [exp(-β·F(x))]^(1/α)
    Weights: w_k ∝ exp(-β/α · (f_k - b))

    Special cases:
    - α → 1: KL divergence
    - α = 2: χ² divergence
    - α → ∞: max divergence

    Properties:
    - Interpolates between different divergences
    - α < 1: more exploratory
    - α > 1: more exploitative
    """

    def __init__(self, alpha: float = 2.0):
        """
        Args:
            alpha: Rényi parameter (α > 0, α ≠ 1)
        """
        assert alpha > 0 and alpha != 1.0, "Rényi alpha must be > 0 and ≠ 1"
        self.alpha = alpha

    def compute_weights(self, f_values: torch.Tensor, beta: float, baseline: float) -> torch.Tensor:
        advantages = f_values - baseline

        # Effective temperature scaled by alpha
        # w ∝ exp(-A/(β·α)) = exp(-A/effective_beta)
        # FIX: was -effective_beta * advantages, now -advantages / effective_beta
        effective_beta = beta * self.alpha  # Note: multiply not divide
        log_weights = -advantages / effective_beta

        # Log-sum-exp trick
        max_log_w = log_weights.max()
        log_weights_stable = log_weights - max_log_w
        weights = torch.exp(log_weights_stable)
        weights = weights / weights.sum()

        # Handle numerical issues
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            weights = torch.ones_like(weights) / len(weights)

        return weights

    def name(self) -> str:
        return f"Renyi(α={self.alpha})"


class TsallisDivergence(Divergence):
    """
    Tsallis divergence: D_q(π || π_ref) = (1/(q-1)) E[(π/π_ref)^q - 1].

    Optimal policy: π*(x) ∝ π_ref(x) · [1 - (1-q)·β·F(x)]^(1/(1-q))

    Special cases:
    - q → 1: KL divergence
    - q = 2: related to χ² divergence

    Properties:
    - Non-extensive entropy
    - Heavy-tailed distributions when q < 1
    - Light-tailed distributions when q > 1
    """

    def __init__(self, q: float = 2.0):
        """
        Args:
            q: Tsallis parameter (q > 0, q ≠ 1)
        """
        assert q > 0 and q != 1.0, "Tsallis q must be > 0 and ≠ 1"
        self.q = q

    def compute_weights(self, f_values: torch.Tensor, beta: float, baseline: float) -> torch.Tensor:
        advantages = f_values - baseline

        # Compute weights: w_k ∝ [1 - (1-q)·A_k/β]^(1/(1-q))
        # FIX: was q_factor * beta * advantages, now q_factor * advantages / beta
        q_factor = 1.0 - self.q
        inside = 1.0 - q_factor * advantages / beta

        # Clamp to positive values
        inside = torch.clamp(inside, min=1e-8)

        # Power transformation
        power = 1.0 / q_factor
        weights = torch.pow(inside, power)

        # Normalize
        weights_sum = weights.sum()
        if weights_sum > 1e-8:
            weights = weights / weights_sum
        else:
            weights = torch.ones_like(weights) / len(weights)

        return weights

    def name(self) -> str:
        return f"Tsallis(q={self.q})"


class HuberDivergence(Divergence):
    """
    Huber-like divergence: hybrid between KL and quadratic.

    Uses KL for small deviations, quadratic for large deviations.

    Weights:
    - |A_k| ≤ δ: w_k ∝ exp(-β·A_k)  (KL regime)
    - |A_k| > δ: w_k ∝ exp(-β·δ·sign(A_k)) (saturated)

    Properties:
    - Robust to outliers (large function values)
    - Combines benefits of KL and χ²
    - Threshold δ controls transition
    """

    def __init__(self, delta: float = 1.0):
        """
        Args:
            delta: Threshold for switching between KL and quadratic
        """
        assert delta > 0, "Huber delta must be positive"
        self.delta = delta

    def compute_weights(self, f_values: torch.Tensor, beta: float, baseline: float) -> torch.Tensor:
        advantages = f_values - baseline

        # Huber-clipped advantages
        huber_advantages = torch.where(
            torch.abs(advantages) <= self.delta,
            advantages,
            self.delta * torch.sign(advantages)
        )

        log_weights = -beta * huber_advantages

        # Log-sum-exp trick
        max_log_w = log_weights.max()
        log_weights_stable = log_weights - max_log_w
        weights = torch.exp(log_weights_stable)
        weights = weights / weights.sum()

        # Handle numerical issues
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            weights = torch.ones_like(weights) / len(weights)

        return weights

    def name(self) -> str:
        return f"Huber(δ={self.delta})"


# Factory function
def get_divergence(name: str, **kwargs) -> Divergence:
    """
    Factory function to create divergence objects.

    Args:
        name: Divergence name ('kl', 'reverse_kl', 'chi2', 'renyi', 'tsallis', 'huber')
        **kwargs: Additional parameters for specific divergences

    Returns:
        Divergence object

    Examples:
        >>> div = get_divergence('kl')
        >>> div = get_divergence('renyi', alpha=2.0)
        >>> div = get_divergence('huber', delta=1.5)
    """
    name_lower = name.lower()

    if name_lower == 'kl':
        return KLDivergence()
    elif name_lower in ['reverse_kl', 'reversekl']:
        return ReverseKLDivergence()
    elif name_lower in ['chi2', 'chisquared', 'chi_squared']:
        return ChiSquaredDivergence()
    elif name_lower == 'renyi':
        alpha = kwargs.get('alpha', 2.0)
        return RenyiDivergence(alpha=alpha)
    elif name_lower == 'tsallis':
        q = kwargs.get('q', 2.0)
        return TsallisDivergence(q=q)
    elif name_lower == 'huber':
        delta = kwargs.get('delta', 1.0)
        return HuberDivergence(delta=delta)
    else:
        raise ValueError(f"Unknown divergence: {name}. "
                        f"Available: 'kl', 'reverse_kl', 'chi2', 'renyi', 'tsallis', 'huber'")
