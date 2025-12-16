"""
AdaSmoothES: Combining KL-regularized moment matching with Evolution Strategy stability.

Theory:
- Boltzmann-weighted moment matching (AdaSmooth framework)
- Diagonal covariance with additive updates (SepCMA stability)
- Evolution path for cumulative learning
- CSA for step-size adaptation
- Baseline for variance reduction

Author: Based on theoretical framework in /home/zlouyang/ZoAR/Docx/AdaSepCMA.md
"""

import math
import torch
from typing import Iterator


class AdaSmoothES(torch.optim.Optimizer):
    """
    AdaSmoothES: KL-regularized moment matching + Evolution Strategy stability.

    Features:
    - Boltzmann weights with baseline for numerical stability
    - Diagonal covariance with additive updates
    - Evolution path for cumulative learning
    - CSA for step-size adaptation

    Theoretical foundation:
    - Theorem 1: Boltzmann optimal policy
    - Theorem 2: Diagonal Gaussian projection (moment matching)
    - Theorem 4: Stabilized update with temporal smoothing
    - Theorem 5: Evolution path as sufficient statistic
    - Theorem 6: Baseline for variance reduction
    """

    def __init__(
        self,
        params: Iterator[torch.Tensor],
        sigma: float = 0.1,
        num_queries: int = 10,
        beta_init: float = 10.0,
        beta_decay: float = 0.001,
        beta_schedule: str = 'polynomial',
        baseline: str = 'mean',  # 'mean', 'min', 'ema', 'none'
        ema_alpha: float = 0.1,  # For EMA baseline
    ):
        """
        Args:
            params: Parameters to optimize
            sigma: Initial step size
            num_queries: Population size K
            beta_init: Initial temperature
            beta_decay: Temperature decay rate
            beta_schedule: 'constant', 'exponential', 'polynomial'
            baseline: Baseline type for variance reduction
            ema_alpha: EMA coefficient for baseline (if baseline='ema')
        """
        defaults = dict(sigma=sigma)
        super().__init__(params, defaults)

        self.K = num_queries
        self.sigma = sigma
        self.beta_init = beta_init
        self.beta_decay = beta_decay
        self.beta_schedule = beta_schedule
        self.baseline_type = baseline
        self.ema_alpha = ema_alpha
        self.iteration = 0

        # EMA baseline state
        self.baseline_ema = None

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
            'path_norm': [], 'baseline': []
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
        """Temperature schedule"""
        t = self.iteration
        if self.beta_schedule == 'constant':
            return self.beta_init
        elif self.beta_schedule == 'exponential':
            return self.beta_init * math.exp(-self.beta_decay * t)
        else:  # polynomial
            return self.beta_init / (1.0 + self.beta_decay * t)

    def _compute_baseline(self, f_values: torch.Tensor) -> float:
        """
        Compute baseline for variance reduction (Theorem 6).

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
        Compute Boltzmann weights with baseline (Theorem 6).

        w_k = softmax(-(f_k - b) / β)

        Uses log-sum-exp trick for numerical stability.
        """
        # Compute baseline
        b = self._compute_baseline(f_values)

        # Advantage: A_k = f_k - b
        advantages = f_values - b

        # Log weights with numerical stability
        # log w_k = -A_k/β - logsumexp(-A/β)
        log_weights = -advantages / beta

        # Log-sum-exp trick
        max_log_w = log_weights.max()
        log_weights_stable = log_weights - max_log_w
        weights = torch.exp(log_weights_stable)
        weights = weights / weights.sum()

        # Handle numerical issues
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            weights = torch.ones_like(weights) / len(weights)

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

        Implements:
        - Theorem 2: Moment matching for mean and variance
        - Theorem 4: Stabilized additive covariance update
        - Theorem 5: Evolution path
        - Theorem 6: Baseline variance reduction
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

        # ===== 2. Weights with Baseline (Theorem 6) =====
        W, advantages, baseline = self._compute_weights(Y, beta_t)

        # Effective mu_eff
        mu_eff_actual = 1.0 / (W ** 2).sum().item()

        # ===== 3. Mean Update (Theorem 2: Moment Matching) =====
        theta_new = torch.sum(W.unsqueeze(1) * X, dim=0)

        # ===== 4. Evolution Paths (Theorem 5) =====
        y_w = (theta_new - theta_t) / self.sigma

        # pc: for covariance
        pc_new = (1 - self.cc) * pc_t + math.sqrt(self.cc * (2 - self.cc) * mu_eff_actual) * y_w

        # p_sigma: for step-size (isotropic)
        p_sigma_new = (1 - self.c_sigma) * p_sigma_t + \
                      math.sqrt(self.c_sigma * (2 - self.c_sigma) * mu_eff_actual) * (y_w / torch.sqrt(c_t + 1e-16))

        # ===== 5. Covariance Update (Theorem 4: Stabilized Moment Matching) =====
        Y_samples = (X - theta_t.unsqueeze(0)) / self.sigma

        # Rank-one: history moment (Theorem 5)
        rank_one = pc_new ** 2

        # Rank-mu: current moment matching (Theorem 2)
        rank_mu = torch.sum(W.unsqueeze(1) * (Y_samples ** 2), dim=0)

        # Additive update (Theorem 4)
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

        self.iteration += 1

        return torch.sum(W * Y).item()
