"""
AdaSmoothES v2: Modular version with pluggable divergences and temperature schedules.

This version extends the original AdaSmoothES (adasmooth_es.py) with:
1. Support for different divergence functionals (KL, Reverse KL, χ², Rényi, etc.)
2. Support for different temperature scheduling strategies
3. Modular design for easy experimentation

Key differences from v1:
- v1: Hard-coded KL divergence + polynomial temperature schedule
- v2: Pluggable divergence and schedule objects

Author: Extended from original AdaSmoothES implementation
"""

import math
import torch
from typing import Iterator, Optional, Union

from optimizer.divergences import Divergence, get_divergence
from optimizer.temperature_schedules import TemperatureSchedule, get_temperature_schedule


class AdaSmoothESv2(torch.optim.Optimizer):
    """
    AdaSmoothES v2: Modular KL-regularized ES with SepCMA stability.

    Solves: min E[F(x)] + (1/β) * D(π || π_ref)

    Where:
    - F(x): objective function
    - D: divergence functional (pluggable)
    - β: temperature (scheduled)
    - π: search distribution N(θ, σ²·diag(c))

    Features:
    - Modular divergence (KL, Reverse KL, χ², Rényi, Tsallis, Huber)
    - Modular temperature scheduling (constant, linear, exponential, etc.)
    - Diagonal covariance with additive updates (SepCMA-style)
    - Evolution paths for cumulative learning
    - CSA for step-size adaptation
    - Baseline for variance reduction

    Theoretical foundation: See /home/zlouyang/ZoAR/Docx/AdaSepCMA.md
    """

    def __init__(
        self,
        params: Iterator[torch.Tensor],
        sigma: float = 0.1,
        num_queries: int = 10,
        divergence: Union[str, Divergence] = 'kl',
        temperature_schedule: Union[str, TemperatureSchedule] = 'polynomial',
        baseline: str = 'mean',
        ema_alpha: float = 0.1,
        # Divergence parameters
        divergence_kwargs: Optional[dict] = None,
        # Temperature schedule parameters
        temperature_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            params: Parameters to optimize
            sigma: Initial step size
            num_queries: Population size K
            divergence: Divergence type (str or Divergence object)
                       Options: 'kl', 'reverse_kl', 'chi2', 'renyi', 'tsallis', 'huber'
            temperature_schedule: Temperature schedule (str or TemperatureSchedule object)
                                 Options: 'constant', 'linear', 'exponential', 'polynomial',
                                         'cosine', 'step', 'adaptive', 'cyclic'
            baseline: Baseline type for variance reduction ('mean', 'min', 'ema', 'none')
            ema_alpha: EMA coefficient for baseline (if baseline='ema')
            divergence_kwargs: Additional kwargs for divergence (e.g., alpha for Rényi)
            temperature_kwargs: Additional kwargs for temperature schedule

        Examples:
            >>> # KL divergence with polynomial schedule (default)
            >>> optimizer = AdaSmoothESv2(params)

            >>> # Rényi divergence with cosine annealing
            >>> optimizer = AdaSmoothESv2(
            ...     params,
            ...     divergence='renyi',
            ...     divergence_kwargs={'alpha': 2.0},
            ...     temperature_schedule='cosine',
            ...     temperature_kwargs={'beta_init': 10.0, 'beta_min': 0.1, 'total_iterations': 10000}
            ... )

            >>> # Chi-squared divergence with adaptive temperature
            >>> optimizer = AdaSmoothESv2(
            ...     params,
            ...     divergence='chi2',
            ...     temperature_schedule='adaptive'
            ... )
        """
        defaults = dict(sigma=sigma)
        super().__init__(params, defaults)

        self.K = num_queries
        self.sigma = sigma
        self.baseline_type = baseline
        self.ema_alpha = ema_alpha
        self.iteration = 0

        # EMA baseline state
        self.baseline_ema = None

        # ===== Divergence setup =====
        if isinstance(divergence, str):
            div_kwargs = divergence_kwargs or {}
            self.divergence = get_divergence(divergence, **div_kwargs)
        else:
            self.divergence = divergence

        # ===== Temperature schedule setup =====
        if isinstance(temperature_schedule, str):
            temp_kwargs = temperature_kwargs or {}
            # Set defaults if not provided
            if 'beta_init' not in temp_kwargs:
                temp_kwargs['beta_init'] = 10.0
            if temperature_schedule == 'polynomial' and 'decay_rate' not in temp_kwargs:
                temp_kwargs['decay_rate'] = 0.001
            if temperature_schedule in ['exponential', 'polynomial', 'linear', 'cosine'] and 'beta_min' not in temp_kwargs:
                temp_kwargs['beta_min'] = 0.01

            self.temperature_schedule = get_temperature_schedule(temperature_schedule, **temp_kwargs)
        else:
            self.temperature_schedule = temperature_schedule

        # Compute total dimension
        self.dim = sum(p.numel() for group in self.param_groups for p in group['params'])
        d = self.dim

        # ===== SepCMA-style hyperparameters =====
        self.mu_eff = num_queries / 2.0

        # Learning rates
        self.cc = 4.0 / (d + 4.0)
        self.c_sigma = (self.mu_eff + 2.0) / (d + self.mu_eff + 3.0)
        self.c1 = 2.0 / ((d + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((d + 2.0) ** 2 + self.mu_eff)
        )

        # Damping
        self.d_sigma = 1.0 + 2.0 * max(0, math.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma

        # Expected norm
        self.chi_d = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d ** 2))

        # Initialize state
        self._initialize_state()

        # History
        self.history = {
            'f_values': [], 'advantages': [], 'weights': [],
            'beta': [], 'sigma': [], 'c_mean': [],
            'path_norm': [], 'baseline': [],
            'divergence': [], 'temperature_name': []
        }

    def _initialize_state(self):
        """Initialize diagonal covariance and evolution paths"""
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                d = param.numel()
                state['c'] = torch.ones(d, device=param.device, dtype=param.dtype)
                state['pc'] = torch.zeros(d, device=param.device, dtype=param.dtype)
                state['p_sigma'] = torch.zeros(d, device=param.device, dtype=param.dtype)

    def _get_beta(self) -> float:
        """Get current temperature from schedule"""
        beta = self.temperature_schedule.get_temperature(self.iteration)

        # Update adaptive schedule if needed
        if hasattr(self.temperature_schedule, 'update') and len(self.history['f_values']) > 0:
            recent_loss = self.history['f_values'][-1].min()
            self.temperature_schedule.update(recent_loss)

        return beta

    def _compute_baseline(self, f_values: torch.Tensor) -> float:
        """
        Compute baseline for variance reduction.

        Args:
            f_values: Function values tensor of shape (K,)

        Returns:
            Baseline value b
        """
        if self.baseline_type == 'none':
            return 0.0
        elif self.baseline_type == 'mean':
            return f_values.mean().item()
        elif self.baseline_type == 'min':
            return f_values.min().item()
        elif self.baseline_type == 'ema':
            current_mean = f_values.mean().item()
            if self.baseline_ema is None:
                self.baseline_ema = current_mean
            else:
                self.baseline_ema = (1 - self.ema_alpha) * self.baseline_ema + self.ema_alpha * current_mean
            return self.baseline_ema
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")

    def _compute_weights(self, f_values: torch.Tensor, beta: float):
        """
        Compute importance weights using the specified divergence.

        Args:
            f_values: Function values of shape (K,)
            beta: Temperature parameter

        Returns:
            weights: Normalized importance weights
            advantages: Advantages (f - baseline)
            baseline: Baseline value
        """
        # Compute baseline
        b = self._compute_baseline(f_values)

        # Compute advantages
        advantages = f_values - b

        # Use divergence to compute weights
        weights = self.divergence.compute_weights(f_values, beta, b)

        return weights, advantages, b

    def _flatten_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for group in self.param_groups for p in group['params']])

    def _unflatten_to_params(self, flat: torch.Tensor):
        offset = 0
        for group in self.param_groups:
            for param in group['params']:
                numel = param.numel()
                param.data = flat[offset:offset + numel].view_as(param)
                offset += numel

    def _get_state_vectors(self):
        c_list, pc_list, p_sigma_list = [], [], []
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                c_list.append(state['c'])
                pc_list.append(state['pc'])
                p_sigma_list.append(state['p_sigma'])
        return torch.cat(c_list), torch.cat(pc_list), torch.cat(p_sigma_list)

    def _set_state_vectors(self, c: torch.Tensor, pc: torch.Tensor, p_sigma: torch.Tensor):
        offset = 0
        for group in self.param_groups:
            for param in group['params']:
                numel = param.numel()
                state = self.state[param]
                state['c'] = c[offset:offset + numel]
                state['pc'] = pc[offset:offset + numel]
                state['p_sigma'] = p_sigma[offset:offset + numel]
                offset += numel

    @torch.no_grad()
    def step(self, closure):
        """
        Perform one optimization step.

        Implements the AdaSmoothES update with pluggable divergence and temperature.

        Args:
            closure: Function that evaluates the model and returns loss

        Returns:
            Weighted average loss
        """
        assert closure is not None, "Closure required"

        beta_t = self._get_beta()

        # Current state
        theta_t = self._flatten_params()
        c_t, pc_t, p_sigma_t = self._get_state_vectors()
        d = theta_t.shape[0]
        device = theta_t.device
        dtype = theta_t.dtype

        # Standard deviations
        std = self.sigma * torch.sqrt(c_t)

        # ===== 1. Sampling =====
        X = []
        Y = []
        Z = []

        for k in range(self.K):
            z_k = torch.randn(d, device=device, dtype=dtype)
            x_k = theta_t + std * z_k

            self._unflatten_to_params(x_k)
            f_val = closure()
            if isinstance(f_val, torch.Tensor):
                f_val = f_val.item()

            X.append(x_k)
            Y.append(f_val)
            Z.append(z_k)

        X = torch.stack(X)
        Y = torch.tensor(Y, device=device, dtype=dtype)
        Z = torch.stack(Z)

        # ===== 2. Weights with Divergence =====
        W, advantages, baseline = self._compute_weights(Y, beta_t)

        # Effective mu_eff
        mu_eff_actual = 1.0 / (W ** 2).sum().item()

        # ===== 3. Mean Update (Moment Matching) =====
        theta_new = torch.sum(W.unsqueeze(1) * X, dim=0)

        # ===== 4. Evolution Paths =====
        y_w = (theta_new - theta_t) / self.sigma

        # pc: for covariance
        pc_new = (1 - self.cc) * pc_t + math.sqrt(self.cc * (2 - self.cc) * mu_eff_actual) * y_w

        # p_sigma: for step-size (isotropic)
        p_sigma_new = (1 - self.c_sigma) * p_sigma_t + \
                      math.sqrt(self.c_sigma * (2 - self.c_sigma) * mu_eff_actual) * (y_w / torch.sqrt(c_t + 1e-16))

        # ===== 5. Covariance Update (Stabilized Moment Matching) =====
        Y_samples = (X - theta_t.unsqueeze(0)) / self.sigma

        # Rank-one: history moment
        rank_one = pc_new ** 2

        # Rank-mu: current moment matching
        rank_mu = torch.sum(W.unsqueeze(1) * (Y_samples ** 2), dim=0)

        # Additive update
        c_new = (1 - self.c1 - self.cmu) * c_t + self.c1 * rank_one + self.cmu * rank_mu
        c_new = torch.clamp(c_new, min=1e-16)

        # ===== 6. Step-size Adaptation (CSA) =====
        p_sigma_norm = torch.norm(p_sigma_new).item()
        sigma_new = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_d - 1)
        )
        sigma_new = max(1e-16, min(sigma_new, 1e8))

        # ===== 7. Apply Updates =====
        self._unflatten_to_params(theta_new)
        self._set_state_vectors(c_new, pc_new, p_sigma_new)
        self.sigma = sigma_new

        # ===== 8. History =====
        self.history['f_values'].append(Y.cpu().numpy())
        self.history['advantages'].append(advantages.cpu().numpy())
        self.history['weights'].append(W.cpu().numpy())
        self.history['beta'].append(beta_t)
        self.history['sigma'].append(sigma_new)
        self.history['c_mean'].append(c_new.mean().item())
        self.history['path_norm'].append(p_sigma_norm)
        self.history['baseline'].append(baseline)
        self.history['divergence'].append(self.divergence.name())
        self.history['temperature_name'].append(self.temperature_schedule.name())

        self.iteration += 1

        return torch.sum(W * Y).item()

    def get_info(self) -> dict:
        """
        Get information about current configuration.

        Returns:
            Dictionary with divergence, schedule, and parameter info
        """
        return {
            'divergence': self.divergence.name(),
            'temperature_schedule': self.temperature_schedule.name(),
            'current_beta': self._get_beta(),
            'current_sigma': self.sigma,
            'iteration': self.iteration,
            'num_queries': self.K,
            'dimension': self.dim,
            'baseline_type': self.baseline_type,
        }
