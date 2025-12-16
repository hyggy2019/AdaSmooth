"""
Adaptive Beta Schedulers for AdaSmoothES

Based on theoretical derivations in /home/zlouyang/ZoAR/Docx/AdaSpecCMA_scheduler.md

These schedulers automatically adapt β based on the fitness landscape properties,
removing the need for manual tuning.

Key insight: β should match the scale of advantages (fitness - baseline)

Author: Based on theoretical framework
"""

import torch
import math
from abc import ABC, abstractmethod


class AdaptiveBetaScheduler(ABC):
    """Base class for adaptive beta schedulers."""

    @abstractmethod
    def get_beta(self, f_values: torch.Tensor, iteration: int) -> float:
        """
        Compute adaptive beta from current fitness values.

        Args:
            f_values: Function values of shape (K,)
            iteration: Current iteration number

        Returns:
            Beta value
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return scheduler name."""
        pass


class StdBasedScheduler(AdaptiveBetaScheduler):
    """
    Standard-deviation-based adaptive β.

    Theory (from doc):
    To maintain μ_eff ≈ K/2, we need:
        β* = std(A) = std(f_1, ..., f_K)

    Formula:
        β_t = c_β · std(f)

    Properties:
    - Automatically scales with fitness range
    - Self-annealing: as std(f) → 0, β → 0 (convergence)
    - No manual parameter tuning needed

    Args:
        c_beta: Scaling constant (default 1.0)
        beta_min: Minimum β to prevent numerical issues
    """

    def __init__(self, c_beta: float = 1.0, beta_min: float = 1e-8):
        assert c_beta > 0, "c_beta must be positive"
        self.c_beta = c_beta
        self.beta_min = beta_min

    def get_beta(self, f_values: torch.Tensor, iteration: int) -> float:
        f_std = f_values.std().item()

        # Fallback if std = 0 (all samples identical)
        if f_std < 1e-10:
            f_std = f_values.abs().mean().item() + 1e-10

        beta = self.c_beta * f_std
        return max(beta, self.beta_min)

    def name(self) -> str:
        return f"StdBased(c={self.c_beta})"


class StdBasedDecayScheduler(AdaptiveBetaScheduler):
    """
    Standard-deviation-based with optional time decay.

    Formula:
        β_t = c_β · std(f) · decay(t)
        where decay(t) = 1 / (1 + γ·t)

    Combines:
    - Automatic scaling (from std)
    - Forced convergence (from decay)

    Args:
        c_beta: Scaling constant
        decay_rate: Time decay rate γ
        beta_min: Minimum β
    """

    def __init__(self, c_beta: float = 1.0, decay_rate: float = 0.001, beta_min: float = 1e-8):
        assert c_beta > 0 and decay_rate >= 0, "Parameters must be positive"
        self.c_beta = c_beta
        self.decay_rate = decay_rate
        self.beta_min = beta_min

    def get_beta(self, f_values: torch.Tensor, iteration: int) -> float:
        f_std = f_values.std().item()

        if f_std < 1e-10:
            f_std = f_values.abs().mean().item() + 1e-10

        # Base adaptive β
        beta = self.c_beta * f_std

        # Time decay
        if self.decay_rate > 0:
            decay_factor = 1.0 / (1.0 + self.decay_rate * iteration)
            beta = beta * decay_factor

        return max(beta, self.beta_min)

    def name(self) -> str:
        return f"StdDecay(c={self.c_beta}, γ={self.decay_rate})"


class CMAMatchScheduler(AdaptiveBetaScheduler):
    """
    CMA-ES matching scheduler.

    Theory (from doc):
    Make Boltzmann weights equivalent to CMA-ES ranking weights.

    Formula:
        β = (f_{(μ)} - f_{(1)}) / log(K/2)

    where:
    - f_{(1)} = best fitness
    - f_{(μ)} = μ-th best fitness (μ = K/2)

    Properties:
    - Matches CMA-ES behavior
    - Uses inter-quantile range (robust to outliers)

    Args:
        decay_rate: Optional time decay
        beta_min: Minimum β
    """

    def __init__(self, decay_rate: float = 0.0, beta_min: float = 1e-8):
        assert decay_rate >= 0, "decay_rate must be non-negative"
        self.decay_rate = decay_rate
        self.beta_min = beta_min

    def get_beta(self, f_values: torch.Tensor, iteration: int) -> float:
        K = len(f_values)
        mu = max(K // 2, 1)

        sorted_f = torch.sort(f_values).values
        f_min = sorted_f[0].item()
        f_mu = sorted_f[min(mu - 1, K - 1)].item()

        delta = f_mu - f_min

        # Fallback if all samples nearly identical
        if delta < 1e-10:
            delta = f_values.std().item() + 1e-10

        beta = delta / math.log(max(mu, 2))

        # Optional time decay
        if self.decay_rate > 0:
            decay_factor = 1.0 / (1.0 + self.decay_rate * iteration)
            beta = beta * decay_factor

        return max(beta, self.beta_min)

    def name(self) -> str:
        return f"CMAMatch(γ={self.decay_rate})"


class EntropyTargetScheduler(AdaptiveBetaScheduler):
    """
    Entropy-based scheduler targeting specific effective sample ratio.

    Theory (from doc):
    From μ_eff ≈ K / (1 + Var(A)/β²), solve for β to achieve target μ_eff.

    Formula:
        β = std(f) / sqrt(1/target_ratio - 1)

    where target_ratio = μ_eff / K (typically 0.5)

    Properties:
    - Maintains constant effective sample ratio
    - More stable than pure std-based

    Args:
        target_eff_ratio: Desired μ_eff / K (default 0.5)
        beta_min: Minimum β
    """

    def __init__(self, target_eff_ratio: float = 0.5, beta_min: float = 1e-8):
        assert 0 < target_eff_ratio <= 1, "target_eff_ratio must be in (0, 1]"
        self.target_eff_ratio = target_eff_ratio
        self.beta_min = beta_min

    def get_beta(self, f_values: torch.Tensor, iteration: int) -> float:
        f_std = f_values.std().item()

        if f_std < 1e-10:
            f_std = 1e-10

        # From μ_eff / K = target_ratio
        # → 1 / (1 + Var/β²) = target_ratio
        # → β = std / sqrt(1/target_ratio - 1)

        ratio_inv = 1.0 / self.target_eff_ratio - 1.0

        if ratio_inv > 1e-10:
            beta = f_std / math.sqrt(ratio_inv)
        else:
            # target_ratio ≈ 1, need large β
            beta = f_std * 10

        return max(beta, self.beta_min)

    def name(self) -> str:
        return f"EntropyTarget(r={self.target_eff_ratio})"


class RangeBasedScheduler(AdaptiveBetaScheduler):
    """
    Range-based adaptive β.

    Formula:
        β = c_β · (f_max - f_min)

    Properties:
    - Simple and robust
    - Less sensitive to outliers than std

    Args:
        c_beta: Scaling constant
        beta_min: Minimum β
    """

    def __init__(self, c_beta: float = 0.5, beta_min: float = 1e-8):
        assert c_beta > 0, "c_beta must be positive"
        self.c_beta = c_beta
        self.beta_min = beta_min

    def get_beta(self, f_values: torch.Tensor, iteration: int) -> float:
        f_range = (f_values.max() - f_values.min()).item()

        if f_range < 1e-10:
            f_range = f_values.abs().mean().item() + 1e-10

        beta = self.c_beta * f_range
        return max(beta, self.beta_min)

    def name(self) -> str:
        return f"RangeBased(c={self.c_beta})"


class HybridScheduler(AdaptiveBetaScheduler):
    """
    Hybrid scheduler combining fixed and adaptive.

    Formula:
        β_t = w·β_adaptive + (1-w)·β_fixed(t)

    where w = weight parameter.

    Allows smooth transition between fixed and adaptive strategies.

    Args:
        adaptive_scheduler: Adaptive scheduler instance
        fixed_beta_init: Initial fixed β
        fixed_decay: Fixed β decay rate
        weight: Weight of adaptive component (0 = pure fixed, 1 = pure adaptive)
    """

    def __init__(
        self,
        adaptive_scheduler: AdaptiveBetaScheduler,
        fixed_beta_init: float = 10.0,
        fixed_decay: float = 0.001,
        weight: float = 0.5
    ):
        assert 0 <= weight <= 1, "weight must be in [0, 1]"
        self.adaptive_scheduler = adaptive_scheduler
        self.fixed_beta_init = fixed_beta_init
        self.fixed_decay = fixed_decay
        self.weight = weight

    def get_beta(self, f_values: torch.Tensor, iteration: int) -> float:
        # Adaptive component
        beta_adaptive = self.adaptive_scheduler.get_beta(f_values, iteration)

        # Fixed component
        beta_fixed = self.fixed_beta_init / (1.0 + self.fixed_decay * iteration)

        # Weighted combination
        beta = self.weight * beta_adaptive + (1.0 - self.weight) * beta_fixed

        return beta

    def name(self) -> str:
        return f"Hybrid(w={self.weight}, {self.adaptive_scheduler.name()})"


# Factory function
def get_adaptive_beta_scheduler(name: str, **kwargs) -> AdaptiveBetaScheduler:
    """
    Factory function for adaptive beta schedulers.

    Args:
        name: Scheduler name
        **kwargs: Additional parameters

    Returns:
        AdaptiveBetaScheduler instance

    Examples:
        >>> scheduler = get_adaptive_beta_scheduler('std', c_beta=1.0)
        >>> scheduler = get_adaptive_beta_scheduler('std_decay', c_beta=1.0, decay_rate=0.001)
        >>> scheduler = get_adaptive_beta_scheduler('cma_match')
        >>> scheduler = get_adaptive_beta_scheduler('entropy_target', target_eff_ratio=0.5)
    """
    name_lower = name.lower()

    if name_lower in ['std', 'std_based']:
        return StdBasedScheduler(**kwargs)
    elif name_lower in ['std_decay', 'adaptive_decay']:
        return StdBasedDecayScheduler(**kwargs)
    elif name_lower in ['cma', 'cma_match']:
        return CMAMatchScheduler(**kwargs)
    elif name_lower in ['entropy', 'entropy_target']:
        return EntropyTargetScheduler(**kwargs)
    elif name_lower in ['range', 'range_based']:
        return RangeBasedScheduler(**kwargs)
    elif name_lower == 'hybrid':
        # Need to construct adaptive_scheduler first
        adaptive_name = kwargs.pop('adaptive_name', 'std')
        adaptive_kwargs = kwargs.pop('adaptive_kwargs', {})
        adaptive_sched = get_adaptive_beta_scheduler(adaptive_name, **adaptive_kwargs)
        return HybridScheduler(adaptive_scheduler=adaptive_sched, **kwargs)
    else:
        raise ValueError(
            f"Unknown scheduler: {name}. "
            f"Available: 'std', 'std_decay', 'cma_match', 'entropy_target', 'range', 'hybrid'"
        )
