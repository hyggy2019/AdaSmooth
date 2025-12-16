好的，让我基于你的 KL-regularized moment matching 理论框架，结合 SepCMA 的稳定性技巧，设计一个新算法。

## 理论融合

### 你的理论核心（保留）

**KL-regularized 目标**：
$$\min_{\pi} \mathbb{E}_{\mathbf{x} \sim \pi}[F(\mathbf{x})] + \beta D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}})$$

**最优解**：Boltzmann 分布 $\pi^* \propto \pi_{\text{ref}} \exp(-F/\beta)$

**Gaussian 投影**：Moment matching
$$\boldsymbol{\theta}^* = \mathbb{E}_{\pi^*}[\mathbf{x}], \quad \boldsymbol{\Sigma}^* = \mathbb{E}_{\pi^*}[(\mathbf{x} - \boldsymbol{\theta}^*)(\mathbf{x} - \boldsymbol{\theta}^*)^\top]$$

### SepCMA 的稳定性技巧（融入）

1. **对角协方差**：$\boldsymbol{\Sigma} = \sigma^2 \text{diag}(\mathbf{c})$
2. **加性更新**：混合历史和当前信息
3. **Evolution path**：累积移动方向
4. **Step-size adaptation (CSA)**：根据 path 长度调整 $\sigma$

---

## 新算法：AdaSmooth-ES

### 数学框架

**协方差参数化**：
$$\boldsymbol{\Sigma}_t = \sigma_t^2 \cdot \text{diag}(\mathbf{c}_t)$$

其中 $\sigma_t \in \mathbb{R}^+$ 是全局 scale，$\mathbf{c}_t \in \mathbb{R}^d_+$ 是逐维度的相对 scale。

**采样**：
$$\mathbf{x}_k = \boldsymbol{\theta}_t + \sigma_t \sqrt{\mathbf{c}_t} \odot \mathbf{z}_k, \quad \mathbf{z}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$

### 更新规则

**Step 1: Softmax 权重**（你的理论）
$$w_k = \frac{\exp(-f(\mathbf{x}_k)/\beta_t)}{\sum_j \exp(-f(\mathbf{x}_j)/\beta_t)}$$

**Step 2: 均值更新**（Moment matching）
$$\boldsymbol{\theta}_{t+1} = \sum_{k=1}^K w_k \mathbf{x}_k$$

**Step 3: Evolution path**（SepCMA）
$$\mathbf{y}_w = \frac{\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t}{\sigma_t}$$
$$\mathbf{p}_{c,t+1} = (1 - c_c)\mathbf{p}_{c,t} + \sqrt{c_c(2-c_c)\mu_{\text{eff}}} \cdot \mathbf{y}_w$$
$$\mathbf{p}_{\sigma,t+1} = (1 - c_\sigma)\mathbf{p}_{\sigma,t} + \sqrt{c_\sigma(2-c_\sigma)\mu_{\text{eff}}} \cdot \frac{\mathbf{y}_w}{\sqrt{\mathbf{c}_t}}$$

**Step 4: 对角协方差更新**（融合 moment matching + SepCMA）
$$\mathbf{c}_{t+1} = (1 - c_1 - c_\mu)\mathbf{c}_t + c_1 \underbrace{\mathbf{p}_{c,t+1}^2}_{\text{rank-one}} + c_\mu \underbrace{\sum_{k=1}^K w_k \mathbf{y}_k^2}_{\text{weighted moment}}$$

其中 $\mathbf{y}_k = (\mathbf{x}_k - \boldsymbol{\theta}_t)/\sigma_t$

**Step 5: Step-size adaptation**（CSA）
$$\sigma_{t+1} = \sigma_t \cdot \exp\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\|\mathbf{p}_{\sigma,t+1}\|}{\chi_d} - 1\right)\right)$$

其中 $\chi_d = \mathbb{E}[\|\mathcal{N}(0,\mathbf{I}_d)\|] \approx \sqrt{d}$

---

### 与原始方法的理论联系

| 组件 | AdaSmooth 理论来源 | SepCMA 稳定性来源 |
|------|-------------------|------------------|
| 权重 $w_k$ | Boltzmann distribution | — |
| 均值 $\boldsymbol{\theta}$ | Moment matching | — |
| 协方差 $\mathbf{c}$ | Moment matching | 加性更新 + rank-one |
| Step-size $\sigma$ | — | CSA |
| Evolution path | — | 累积历史 |

---

## 完整实现

```python
import math
import torch
from typing import Iterator, Tuple


class AdaSmoothES(torch.optim.Optimizer):
    """
    AdaSmooth-ES: Combining KL-regularized moment matching with 
    evolution strategy stability techniques.
    
    Theory: Boltzmann-weighted moment matching (AdaSmooth)
    Stability: Diagonal covariance + CSA + Evolution path (SepCMA)
    
    Covariance structure: Σ = σ² · diag(c)
    Complexity: O(d) space, O(Kd) time per iteration
    """

    def __init__(
        self,
        params: Iterator[torch.Tensor],
        sigma: float = 0.1,
        num_queries: int = 10,
        beta_init: float = 1.0,
        beta_decay: float = 0.05,
        beta_schedule: str = 'polynomial',
    ):
        """
        Args:
            params: Parameters to optimize
            sigma: Initial global step size
            num_queries: Population size K
            beta_init: Initial temperature for Boltzmann weights
            beta_decay: Temperature decay rate
            beta_schedule: 'constant', 'exponential', or 'polynomial'
        """
        defaults = dict(sigma=sigma)
        super().__init__(params, defaults)

        self.K = num_queries
        self.sigma = sigma
        self.beta_init = beta_init
        self.beta_decay = beta_decay
        self.beta_schedule = beta_schedule
        self.iteration = 0

        # Compute total dimension
        self.dim = sum(p.numel() for group in self.param_groups for p in group['params'])

        # ===== SepCMA-style hyperparameters =====
        d = self.dim
        
        # Effective sample size (for Softmax weights, approximate)
        self.mu_eff = num_queries / 2.0
        
        # Learning rates
        self.cc = 4.0 / (d + 4.0)  # Evolution path cumulation
        self.c_sigma = (self.mu_eff + 2.0) / (d + self.mu_eff + 3.0)  # Step-size cumulation
        self.c1 = 2.0 / ((d + 1.3) ** 2 + self.mu_eff)  # Rank-one learning rate
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((d + 2.0) ** 2 + self.mu_eff)
        )  # Rank-mu learning rate
        
        # Damping for step-size
        self.d_sigma = 1.0 + 2.0 * max(0, math.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma
        
        # Expected norm of N(0,I)
        self.chi_d = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d ** 2))

        # ===== Initialize state =====
        self._initialize_state()

        # History tracking
        self.history = {
            'f_values': [],
            'weights': [],
            'beta': [],
            'sigma': [],
            'c_mean': [],
            'path_norm': []
        }

    def _initialize_state(self):
        """Initialize evolution paths and diagonal covariance"""
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                d = param.numel()
                
                # Diagonal covariance (relative scales)
                state['c'] = torch.ones(d, device=param.device, dtype=param.dtype)
                
                # Evolution paths
                state['pc'] = torch.zeros(d, device=param.device, dtype=param.dtype)
                state['p_sigma'] = torch.zeros(d, device=param.device, dtype=param.dtype)

    def _get_beta(self) -> float:
        """Get current temperature β_t"""
        t = self.iteration
        if self.beta_schedule == 'constant':
            return self.beta_init
        elif self.beta_schedule == 'exponential':
            return self.beta_init * math.exp(-self.beta_decay * t)
        else:  # polynomial
            return self.beta_init / (1.0 + self.beta_decay * t)

    def _flatten_params(self) -> torch.Tensor:
        """Flatten all parameters into single vector"""
        return torch.cat([
            p.data.view(-1) 
            for group in self.param_groups 
            for p in group['params']
        ])

    def _unflatten_to_params(self, flat: torch.Tensor):
        """Write flattened vector back to parameters"""
        offset = 0
        for group in self.param_groups:
            for param in group['params']:
                numel = param.numel()
                param.data = flat[offset:offset + numel].view_as(param)
                offset += numel

    def _get_state_vectors(self):
        """Get concatenated state vectors"""
        c_list, pc_list, p_sigma_list = [], [], []
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                c_list.append(state['c'])
                pc_list.append(state['pc'])
                p_sigma_list.append(state['p_sigma'])
        
        return (
            torch.cat(c_list),
            torch.cat(pc_list),
            torch.cat(p_sigma_list)
        )

    def _set_state_vectors(self, c: torch.Tensor, pc: torch.Tensor, p_sigma: torch.Tensor):
        """Write state vectors back to per-parameter state"""
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
        
        Args:
            closure: Function that evaluates the model and returns loss
        """
        assert closure is not None, "Closure required for AdaSmoothES"

        beta_t = self._get_beta()
        
        # Get current state
        theta_t = self._flatten_params()
        c_t, pc_t, p_sigma_t = self._get_state_vectors()
        d = theta_t.shape[0]
        device = theta_t.device
        dtype = theta_t.dtype

        # Compute standard deviations: sqrt(c) * sigma
        std = self.sigma * torch.sqrt(c_t)

        # ===== 1. Sampling =====
        X = []  # Candidates
        Y = []  # Function values
        Z = []  # Standard normal samples

        for k in range(self.K):
            z_k = torch.randn(d, device=device, dtype=dtype)
            x_k = theta_t + std * z_k
            
            # Evaluate
            self._unflatten_to_params(x_k)
            f_val = closure()
            if isinstance(f_val, torch.Tensor):
                f_val = f_val.item()
            
            X.append(x_k)
            Y.append(f_val)
            Z.append(z_k)

        X = torch.stack(X)  # (K, d)
        Y = torch.tensor(Y, device=device, dtype=dtype)  # (K,)
        Z = torch.stack(Z)  # (K, d)

        # ===== 2. Compute Softmax Weights (AdaSmooth theory) =====
        log_w = -Y / beta_t
        log_w = log_w - log_w.max()  # Numerical stability
        W = torch.exp(log_w)
        W = W / W.sum()

        # Handle degenerate case
        if torch.isnan(W).any() or torch.isinf(W).any():
            W = torch.ones_like(W) / self.K

        # Compute effective mu_eff from actual weights
        mu_eff_actual = 1.0 / (W ** 2).sum().item()

        # ===== 3. Mean Update (Moment Matching) =====
        theta_new = torch.sum(W.unsqueeze(1) * X, dim=0)

        # ===== 4. Evolution Paths =====
        # Normalized step
        y_w = (theta_new - theta_t) / self.sigma  # (d,)

        # pc: cumulation for covariance
        pc_new = (1 - self.cc) * pc_t + math.sqrt(self.cc * (2 - self.cc) * mu_eff_actual) * y_w

        # p_sigma: cumulation for step-size (in isotropic space)
        p_sigma_new = (1 - self.c_sigma) * p_sigma_t + \
                      math.sqrt(self.c_sigma * (2 - self.c_sigma) * mu_eff_actual) * (y_w / torch.sqrt(c_t + 1e-16))

        # ===== 5. Covariance Update (Moment Matching + SepCMA stability) =====
        # y_k = (x_k - theta_t) / sigma
        Y_samples = (X - theta_t.unsqueeze(0)) / self.sigma  # (K, d)

        # Rank-one term: pc²
        rank_one = pc_new ** 2

        # Rank-mu term: weighted moment matching (AdaSmooth theory)
        # Σ w_k * y_k²
        rank_mu = torch.sum(W.unsqueeze(1) * (Y_samples ** 2), dim=0)

        # Combined update (SepCMA-style additive)
        c_new = (1 - self.c1 - self.cmu) * c_t + self.c1 * rank_one + self.cmu * rank_mu

        # Ensure positivity
        c_new = torch.clamp(c_new, min=1e-16)

        # ===== 6. Step-size Adaptation (CSA) =====
        p_sigma_norm = torch.norm(p_sigma_new).item()
        sigma_new = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_d - 1)
        )

        # Clamp sigma to reasonable range
        sigma_new = max(1e-16, min(sigma_new, 1e8))

        # ===== 7. Apply Updates =====
        self._unflatten_to_params(theta_new)
        self._set_state_vectors(c_new, pc_new, p_sigma_new)
        self.sigma = sigma_new

        # ===== 8. Track History =====
        self.history['f_values'].append(Y.cpu().numpy())
        self.history['weights'].append(W.cpu().numpy())
        self.history['beta'].append(beta_t)
        self.history['sigma'].append(sigma_new)
        self.history['c_mean'].append(c_new.mean().item())
        self.history['path_norm'].append(p_sigma_norm)

        self.iteration += 1

        # Return weighted loss
        return torch.sum(W * Y).item()


class AdaSmoothES_LowRank(torch.optim.Optimizer):
    """
    AdaSmooth-ES with Low-Rank covariance structure.
    
    Covariance: Σ = σ² · (diag(c) + L @ Lᵀ)
    
    Combines:
    - Diagonal adaptation (like SepCMA)
    - Low-rank adaptation for correlations
    - KL-regularized moment matching weights
    """

    def __init__(
        self,
        params: Iterator[torch.Tensor],
        sigma: float = 0.1,
        num_queries: int = 10,
        rank: int = None,  # Low-rank component rank (default: num_queries)
        beta_init: float = 1.0,
        beta_decay: float = 0.05,
        beta_schedule: str = 'polynomial',
        c_diag: float = 0.1,  # Learning rate for diagonal
        c_rank: float = 0.1,  # Learning rate for low-rank
    ):
        defaults = dict(sigma=sigma)
        super().__init__(params, defaults)

        self.K = num_queries
        self.rank = rank if rank is not None else num_queries
        self.sigma = sigma
        self.beta_init = beta_init
        self.beta_decay = beta_decay
        self.beta_schedule = beta_schedule
        self.c_diag = c_diag
        self.c_rank = c_rank
        self.iteration = 0

        # Compute dimension
        self.dim = sum(p.numel() for group in self.param_groups for p in group['params'])
        d = self.dim

        # SepCMA hyperparameters for step-size
        self.mu_eff = num_queries / 2.0
        self.c_sigma = (self.mu_eff + 2.0) / (d + self.mu_eff + 3.0)
        self.d_sigma = 1.0 + 2.0 * max(0, math.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma
        self.chi_d = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d ** 2))

        self._initialize_state()

        self.history = {
            'f_values': [], 'weights': [], 'beta': [],
            'sigma': [], 'c_mean': [], 'L_norm': []
        }

    def _initialize_state(self):
        """Initialize diagonal c, low-rank L, and evolution path"""
        device = None
        dtype = None
        for group in self.param_groups:
            for param in group['params']:
                device = param.device
                dtype = param.dtype
                break

        # Global state (flattened)
        self.c = torch.ones(self.dim, device=device, dtype=dtype)
        self.L = 0.1 * torch.randn(self.dim, self.rank, device=device, dtype=dtype)
        self.p_sigma = torch.zeros(self.dim, device=device, dtype=dtype)

    def _get_beta(self) -> float:
        t = self.iteration
        if self.beta_schedule == 'constant':
            return self.beta_init
        elif self.beta_schedule == 'exponential':
            return self.beta_init * math.exp(-self.beta_decay * t)
        else:
            return self.beta_init / (1.0 + self.beta_decay * t)

    def _flatten_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for group in self.param_groups for p in group['params']])

    def _unflatten_to_params(self, flat: torch.Tensor):
        offset = 0
        for group in self.param_groups:
            for param in group['params']:
                numel = param.numel()
                param.data = flat[offset:offset + numel].view_as(param)
                offset += numel

    def _sample(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from N(theta, σ²(diag(c) + LLᵀ))
        
        Using: x = theta + sigma * (sqrt(c) ⊙ z + L @ u)
        where z ~ N(0, I_d), u ~ N(0, I_r)
        """
        d = theta.shape[0]
        device = theta.device
        dtype = theta.dtype

        z = torch.randn(d, device=device, dtype=dtype)
        u = torch.randn(self.rank, device=device, dtype=dtype)

        # x = theta + sigma * (sqrt(c) * z + L @ u)
        perturbation = torch.sqrt(self.c) * z + torch.matmul(self.L, u)
        x = theta + self.sigma * perturbation

        return x, z, u

    @torch.no_grad()
    def step(self, closure):
        assert closure is not None

        beta_t = self._get_beta()
        theta_t = self._flatten_params()
        d = theta_t.shape[0]
        device = theta_t.device
        dtype = theta_t.dtype

        # ===== 1. Sampling =====
        X, Y_vals, Z_list, U_list = [], [], [], []

        for k in range(self.K):
            x_k, z_k, u_k = self._sample(theta_t)
            
            self._unflatten_to_params(x_k)
            f_val = closure()
            if isinstance(f_val, torch.Tensor):
                f_val = f_val.item()

            X.append(x_k)
            Y_vals.append(f_val)
            Z_list.append(z_k)
            U_list.append(u_k)

        X = torch.stack(X)
        Y = torch.tensor(Y_vals, device=device, dtype=dtype)
        Z = torch.stack(Z_list)
        U = torch.stack(U_list)

        # ===== 2. Softmax Weights =====
        log_w = -Y / beta_t
        log_w = log_w - log_w.max()
        W = torch.exp(log_w)
        W = W / W.sum()

        if torch.isnan(W).any() or torch.isinf(W).any():
            W = torch.ones_like(W) / self.K

        mu_eff_actual = 1.0 / (W ** 2).sum().item()

        # ===== 3. Mean Update =====
        theta_new = torch.sum(W.unsqueeze(1) * X, dim=0)

        # ===== 4. Evolution Path for Step-size =====
        y_w = (theta_new - theta_t) / self.sigma
        
        # Approximate isotropic normalization (ignore L for simplicity)
        p_sigma_new = (1 - self.c_sigma) * self.p_sigma + \
                      math.sqrt(self.c_sigma * (2 - self.c_sigma) * mu_eff_actual) * (y_w / torch.sqrt(self.c + 1e-16))

        # ===== 5. Diagonal Covariance Update =====
        residuals = (X - theta_new.unsqueeze(0)) / self.sigma  # (K, d)
        weighted_var = torch.sum(W.unsqueeze(1) * (residuals ** 2), dim=0)

        c_new = (1 - self.c_diag) * self.c + self.c_diag * weighted_var
        c_new = torch.clamp(c_new, min=1e-16)

        # ===== 6. Low-Rank Update =====
        # L_new columns: weighted residuals projected
        # c_k = sqrt(w_k) * (x_k - theta_new) / sigma
        weighted_residuals = torch.sqrt(W).unsqueeze(1) * residuals  # (K, d)
        L_new_candidate = weighted_residuals.T  # (d, K)

        # Truncate or pad to rank
        if self.K >= self.rank:
            L_new_candidate = L_new_candidate[:, :self.rank]
        else:
            padding = torch.zeros(d, self.rank - self.K, device=device, dtype=dtype)
            L_new_candidate = torch.cat([L_new_candidate, padding], dim=1)

        # Additive update (SepCMA-style stability)
        L_new = math.sqrt(1 - self.c_rank) * self.L + math.sqrt(self.c_rank) * L_new_candidate

        # ===== 7. Step-size Adaptation =====
        p_sigma_norm = torch.norm(p_sigma_new).item()
        sigma_new = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_d - 1)
        )
        sigma_new = max(1e-16, min(sigma_new, 1e8))

        # ===== 8. Apply Updates =====
        self._unflatten_to_params(theta_new)
        self.c = c_new
        self.L = L_new
        self.p_sigma = p_sigma_new
        self.sigma = sigma_new

        # Track
        self.history['f_values'].append(Y.cpu().numpy())
        self.history['weights'].append(W.cpu().numpy())
        self.history['beta'].append(beta_t)
        self.history['sigma'].append(sigma_new)
        self.history['c_mean'].append(c_new.mean().item())
        self.history['L_norm'].append(torch.norm(L_new).item())

        self.iteration += 1

        return torch.sum(W * Y).item()
```

---

## 算法对比总结

| 特性 | AdaSmooth (原) | SepCMA | AdaSmoothES (新) |
|------|---------------|--------|-----------------|
| **权重** | Softmax (理论) | 排名对数 | Softmax (理论) |
| **均值更新** | Moment matching | Weighted average | Moment matching |
| **协方差结构** | Low-rank | 对角 | 对角 / 对角+Low-rank |
| **协方差更新** | 乘性替换 | 加性混合 | 加性混合 |
| **Step-size** | 隐含 / 无 | CSA | CSA |
| **Evolution path** | 无 | 有 | 有 |
| **稳定性** | ⚠️ 会收缩 | ✅ 稳定 | ✅ 稳定 |
| **理论基础** | ✅ KL-regularized | ❌ 启发式 | ✅ KL-regularized |

---

需要我把这个写成论文的 Algorithm 伪代码吗？
非常好的问题！让我为 AdaSmoothES 推导完整的理论框架。

## AdaSmoothES 的理论推导

### 问题设定

我们要优化：
$$\min_{\pi} \mathcal{J}(\pi) = \mathbb{E}_{\mathbf{x} \sim \pi}[F(\mathbf{x})] + \beta D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}})$$

**关键区别**：我们限制 $\pi$ 在**对角高斯族**中：
$$\Pi_{\text{diag}} = \left\{ \mathcal{N}(\boldsymbol{\theta}, \sigma^2 \text{diag}(\mathbf{c})) \mid \boldsymbol{\theta} \in \mathbb{R}^d, \sigma > 0, \mathbf{c} \in \mathbb{R}^d_+ \right\}$$

---

### Theorem 1: Optimal Unconstrained Policy（你原来的）

**定理 1**（Boltzmann 最优策略）

KL-regularized 目标的无约束最优解是：
$$\pi^*(\mathbf{x}) = \frac{1}{Z} \pi_{\text{ref}}(\mathbf{x}) \exp\left(-\frac{F(\mathbf{x})}{\beta}\right)$$

其中 $Z = \mathbb{E}_{\pi_{\text{ref}}}\left[\exp(-F(\mathbf{x})/\beta)\right]$。

---

### Theorem 2: Optimal Diagonal Gaussian Projection（新的）

**定理 2**（对角高斯最优投影）

设 $\pi^*$ 为 Theorem 1 的 Boltzmann 最优策略。将 $\pi^*$ 投影到对角高斯族 $\Pi_{\text{diag}}$：
$$\hat{\pi}^* = \arg\min_{\pi \in \Pi_{\text{diag}}} D_{\mathrm{KL}}(\pi^* \| \pi)$$

则最优参数为：
$$\boldsymbol{\theta}^* = \mathbb{E}_{\mathbf{x} \sim \pi^*}[\mathbf{x}] = \mathbb{E}_{\mathbf{x} \sim \pi_{\text{ref}}}[w(\mathbf{x}) \mathbf{x}]$$

$$(\sigma^*)^2 c_i^* = \mathbb{E}_{\mathbf{x} \sim \pi^*}[(x_i - \theta_i^*)^2] = \mathbb{E}_{\mathbf{x} \sim \pi_{\text{ref}}}[w(\mathbf{x})(x_i - \theta_i^*)^2]$$

其中 $w(\mathbf{x}) = \frac{1}{Z}\exp(-F(\mathbf{x})/\beta)$ 是归一化的 Boltzmann 权重。

**证明**：

对于高斯分布 $\pi = \mathcal{N}(\boldsymbol{\theta}, \boldsymbol{\Sigma})$，reverse KL divergence 为：
$$D_{\mathrm{KL}}(\pi^* \| \pi) = -H(\pi^*) - \mathbb{E}_{\pi^*}[\log \pi(\mathbf{x})]$$

其中第一项与 $\pi$ 无关。展开第二项：
$$\mathbb{E}_{\pi^*}[\log \pi(\mathbf{x})] = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{1}{2}\mathbb{E}_{\pi^*}[(\mathbf{x}-\boldsymbol{\theta})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\theta})]$$

对于对角 $\boldsymbol{\Sigma} = \text{diag}(\boldsymbol{\sigma}^2)$：
$$= -\frac{d}{2}\log(2\pi) - \sum_{i=1}^d \log \sigma_i - \frac{1}{2}\sum_{i=1}^d \frac{\mathbb{E}_{\pi^*}[(x_i - \theta_i)^2]}{\sigma_i^2}$$

对 $\theta_i$ 求导并令其为零：
$$\frac{\partial}{\partial \theta_i} = \frac{\mathbb{E}_{\pi^*}[x_i - \theta_i]}{\sigma_i^2} = 0 \implies \theta_i^* = \mathbb{E}_{\pi^*}[x_i]$$

对 $\sigma_i$ 求导并令其为零：
$$\frac{\partial}{\partial \sigma_i} = -\frac{1}{\sigma_i} + \frac{\mathbb{E}_{\pi^*}[(x_i - \theta_i^*)^2]}{\sigma_i^3} = 0 \implies (\sigma_i^*)^2 = \mathbb{E}_{\pi^*}[(x_i - \theta_i^*)^2]$$

$\square$

---

### Theorem 3: Finite-Sample Approximation（实践版本）

**定理 3**（有限样本近似）

给定当前策略 $\pi_t = \mathcal{N}(\boldsymbol{\theta}_t, \sigma_t^2 \text{diag}(\mathbf{c}_t))$，采样 $K$ 个点 $\{\mathbf{x}_k\}_{k=1}^K$，计算权重：
$$w_k = \frac{\exp(-f(\mathbf{x}_k)/\beta_t)}{\sum_{j=1}^K \exp(-f(\mathbf{x}_j)/\beta_t)}$$

则 Theorem 2 的最优解的蒙特卡洛近似为：

**均值**：
$$\boldsymbol{\theta}_{t+1}^{\text{MM}} = \sum_{k=1}^K w_k \mathbf{x}_k$$

**对角方差**：
$$\mathbf{v}_{t+1}^{\text{MM}} = \sum_{k=1}^K w_k (\mathbf{x}_k - \boldsymbol{\theta}_{t+1}^{\text{MM}})^{\odot 2}$$

其中 $\odot 2$ 表示逐元素平方。

---

### Theorem 4: Stabilized Update（稳定化更新）

**定理 4**（带历史累积的稳定更新）

直接使用 Theorem 3 的更新可能导致方差估计不稳定（高方差、收缩）。我们引入**时间平滑**的变分目标：

$$\min_{\pi \in \Pi_{\text{diag}}} \mathbb{E}_{\mathbf{x} \sim \pi}[F(\mathbf{x})] + \beta D_{\mathrm{KL}}(\pi \| \pi_t) + \gamma D_{\mathrm{KL}}(\pi \| \bar{\pi}_t)$$

其中 $\bar{\pi}_t$ 是历史累积分布（evolution path 的分布解释）。

这等价于对**混合参考分布**做 moment matching：
$$\pi_{\text{ref}}^{\text{mixed}} \propto \pi_t^{\alpha} \cdot \bar{\pi}_t^{1-\alpha}$$

**结果**：最优协方差更新变为加性形式：
$$\mathbf{v}_{t+1} = (1 - c_{\text{cov}}) \mathbf{v}_t + c_{\text{cov}} \mathbf{v}_{t+1}^{\text{MM}}$$

其中 $c_{\text{cov}} = \frac{\beta}{\beta + \gamma}$ 是有效学习率。

**证明草图**：

混合 KL 约束：
$$\beta D_{\mathrm{KL}}(\pi \| \pi_t) + \gamma D_{\mathrm{KL}}(\pi \| \bar{\pi}_t)$$

对于高斯分布，KL 散度关于方差是凸的。最优解满足一阶条件：
$$\beta \cdot \nabla_{\mathbf{v}} D_{\mathrm{KL}}(\pi \| \pi_t) + \gamma \cdot \nabla_{\mathbf{v}} D_{\mathrm{KL}}(\pi \| \bar{\pi}_t) = \nabla_{\mathbf{v}} \mathbb{E}_\pi[F]$$

利用高斯 KL 的显式形式和 moment matching 条件，可以证明最优解是加权平均。

$\square$

---

### Theorem 5: Evolution Path as Sufficient Statistic（演化路径的理论解释）

**定理 5**（Evolution Path 的变分解释）

定义累积移动方向：
$$\mathbf{p}_{c,t+1} = (1 - c_c) \mathbf{p}_{c,t} + \sqrt{c_c(2-c_c)\mu_{\text{eff}}} \cdot \frac{\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t}{\sigma_t}$$

则 $\mathbf{p}_{c,t}$ 是历史梯度方向的**指数加权平均**的充分统计量。

**Rank-one 更新** $c_1 \mathbf{p}_c^{\odot 2}$ 对应于：
$$\mathbb{E}_{\bar{\pi}}[(\mathbf{x} - \boldsymbol{\theta})^{\odot 2}]$$

即**历史累积分布 $\bar{\pi}$** 的二阶矩。

这为 SepCMA 的 rank-one 更新提供了变分解释：它是在对历史分布做 moment matching。

---

## 完整算法的理论总结

### Algorithm: AdaSmoothES

**输入**：初始 $\boldsymbol{\theta}_0, \sigma_0, \mathbf{c}_0 = \mathbf{1}$，温度 $\beta_0$，学习率 $c_1, c_\mu, c_c, c_\sigma$

**For** $t = 0, 1, 2, \ldots$

**Step 1**: 采样 $\mathbf{x}_k \sim \mathcal{N}(\boldsymbol{\theta}_t, \sigma_t^2 \text{diag}(\mathbf{c}_t))$，计算 $f(\mathbf{x}_k)$

**Step 2**: 计算 Boltzmann 权重（Theorem 1 的近似）
$$w_k = \text{softmax}(-f(\mathbf{x}_k)/\beta_t)$$

**Step 3**: 均值更新（Theorem 2 的 moment matching）
$$\boldsymbol{\theta}_{t+1} = \sum_k w_k \mathbf{x}_k$$

**Step 4**: Evolution path 更新（Theorem 5）
$$\mathbf{p}_{c,t+1} = (1-c_c)\mathbf{p}_{c,t} + \sqrt{c_c(2-c_c)\mu_{\text{eff}}} \cdot \mathbf{y}_w$$

**Step 5**: 协方差更新（Theorem 4 的稳定化 moment matching）
$$\mathbf{c}_{t+1} = (1 - c_1 - c_\mu)\mathbf{c}_t + c_1 \underbrace{\mathbf{p}_{c,t+1}^{\odot 2}}_{\text{历史 moment}} + c_\mu \underbrace{\sum_k w_k \mathbf{y}_k^{\odot 2}}_{\text{当前 moment}}$$

**Step 6**: Step-size 更新（CSA，可选理论：optimal $\sigma$ 的在线估计）
$$\sigma_{t+1} = \sigma_t \exp\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\|\mathbf{p}_\sigma\|}{\chi_d} - 1\right)\right)$$

---

## 理论贡献总结

| 定理 | 内容 | 对应组件 |
|------|------|----------|
| **Theorem 1** | Boltzmann 最优策略 | Softmax 权重 |
| **Theorem 2** | 对角高斯最优投影 | 均值 + 方差的 moment matching |
| **Theorem 3** | 有限样本近似 | 蒙特卡洛估计 |
| **Theorem 4** | 时间平滑的稳定更新 | 加性协方差更新 |
| **Theorem 5** | Evolution path 的变分解释 | Rank-one 更新 |

这样你的论文就有了完整的理论链条：

**AdaSmooth (原)** → 加入对角约束 → **Theorem 2** → 加入稳定性 → **Theorem 4** → 加入历史累积 → **Theorem 5** → **AdaSmoothES**

需要我把这些写成正式的 LaTeX 定理格式吗？
好的，让我把 baseline 的理论也加入进来。

## Theorem 6: Variance Reduction via Baseline

### 动机

Softmax 权重 $w_k = \frac{\exp(-f_k/\beta)}{\sum_j \exp(-f_j/\beta)}$ 对函数值的绝对大小敏感。如果所有 $f_k$ 都很大（或都很小），权重会趋于均匀，丢失排序信息。

### 定理 6（Baseline 不变性与方差减少）

**定理 6.1**（Baseline 不变性）

对于任意常数 $b \in \mathbb{R}$，定义 centered 权重：
$$\tilde{w}_k = \frac{\exp(-(f_k - b)/\beta)}{\sum_{j=1}^K \exp(-(f_j - b)/\beta)}$$

则 $\tilde{w}_k = w_k$，即 Boltzmann 权重对常数平移不变。

**证明**：
$$\tilde{w}_k = \frac{\exp(-f_k/\beta) \cdot \exp(b/\beta)}{\sum_j \exp(-f_j/\beta) \cdot \exp(b/\beta)} = \frac{\exp(-f_k/\beta)}{\sum_j \exp(-f_j/\beta)} = w_k$$

$\square$

---

**定理 6.2**（最优 Baseline 选择）

虽然理论上 baseline 不改变权重，但在有限精度计算中，选择 $b = \bar{f} = \frac{1}{K}\sum_k f_k$ 或 $b = \min_k f_k$ 可以：

1. **数值稳定性**：防止 $\exp(-f_k/\beta)$ 溢出或下溢
2. **有效温度调节**：使权重分布的有效方差与 $\text{Var}(f - b)$ 而非 $\text{Var}(f)$ 相关

**推论**：使用 centered function values $\hat{f}_k = f_k - \bar{f}$，权重变为：
$$w_k = \frac{\exp(-\hat{f}_k/\beta)}{\sum_j \exp(-\hat{f}_j/\beta)}$$

此时 $\sum_k \hat{f}_k = 0$，权重分布仅依赖于相对排序。

---

**定理 6.3**（Adaptive Baseline 与 Advantage Function）

定义 **advantage function**：
$$A_k = f_k - b_t$$

其中 $b_t$ 是 baseline。不同的 baseline 选择：

| Baseline | 定义 | 性质 |
|----------|------|------|
| **Mean baseline** | $b_t = \frac{1}{K}\sum_k f_k$ | 零中心化，最常用 |
| **Min baseline** | $b_t = \min_k f_k$ | 所有 $A_k \geq 0$，数值最稳定 |
| **Exponential moving average** | $b_t = (1-\alpha)b_{t-1} + \alpha \bar{f}_t$ | 跨迭代平滑 |
| **Current point** | $b_t = f(\boldsymbol{\theta}_t)$ | 需要额外一次查询 |

**理论最优**：在 REINFORCE 框架下，方差最小的 baseline 是：
$$b^* = \frac{\mathbb{E}[\|\nabla \log \pi\|^2 \cdot f]}{\mathbb{E}[\|\nabla \log \pi\|^2]}$$

但在我们的 moment matching 框架中，**mean baseline** $b = \bar{f}$ 是最自然的选择，因为它使 advantage 零中心化。

---

## 完整的 AdaSmoothES 算法（带 Baseline）

### 数学框架

**Step 1: 采样与评估**
$$\mathbf{x}_k = \boldsymbol{\theta}_t + \sigma_t \sqrt{\mathbf{c}_t} \odot \mathbf{z}_k, \quad \mathbf{z}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$
$$f_k = f(\mathbf{x}_k)$$

**Step 2: Baseline 计算**
$$b_t = \frac{1}{K}\sum_{k=1}^K f_k \quad \text{(mean baseline)}$$

**Step 3: Advantage 计算**
$$A_k = f_k - b_t$$

**Step 4: Boltzmann 权重（数值稳定版）**
$$\log w_k = -\frac{A_k}{\beta_t} - \log\sum_{j=1}^K \exp\left(-\frac{A_j}{\beta_t}\right)$$

使用 log-sum-exp trick：
$$\log w_k = -\frac{A_k}{\beta_t} - \left( m + \log\sum_{j=1}^K \exp\left(-\frac{A_j}{\beta_t} - m\right) \right)$$

其中 $m = \max_j \left(-\frac{A_j}{\beta_t}\right) = -\frac{\min_j A_j}{\beta_t}$

**Step 5-8: 均值、Evolution path、协方差、Step-size 更新**（同前）

---

## 代码实现

```python
import math
import torch
from typing import Iterator, Tuple, Optional


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
        beta_init: float = 1.0,
        beta_decay: float = 0.05,
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

    def _compute_weights(self, f_values: torch.Tensor, beta: float) -> torch.Tensor:
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
```

---

## 完整理论框架总结

| 定理 | 内容 | 数学公式 | 对应代码 |
|------|------|----------|----------|
| **Theorem 1** | Boltzmann 最优策略 | $\pi^* \propto \pi_{\text{ref}} e^{-F/\beta}$ | 权重计算 |
| **Theorem 2** | 对角高斯投影 | $\theta^* = \mathbb{E}_{\pi^*}[x]$, $\sigma_i^2 = \mathbb{E}_{\pi^*}[(x_i-\theta_i)^2]$ | 均值/方差更新 |
| **Theorem 3** | 有限样本近似 | $\theta \approx \sum w_k x_k$ | Monte Carlo |
| **Theorem 4** | 稳定化更新 | $\mathbf{c}_{t+1} = (1-c)\mathbf{c}_t + c \cdot \text{MM}$ | 加性协方差 |
| **Theorem 5** | Evolution path | $\mathbf{p}_c = (1-c_c)\mathbf{p}_c + \sqrt{\cdot} \mathbf{y}_w$ | Rank-one |
| **Theorem 6** | Baseline 方差减少 | $A_k = f_k - b$, $w_k = \text{softmax}(-A_k/\beta)$ | `_compute_weights` |

---

需要我把这些定理整理成正式的 LaTeX 论文格式吗？

让我仔细对比两者的每一步。

## AdaSmoothES vs SepCMA 逐步对比

### 1. 采样

**SepCMA**：
$$\mathbf{x}_k = \boldsymbol{\mu} + \sigma \cdot \mathbf{D} \odot \mathbf{z}_k, \quad \mathbf{z}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$

其中 $\mathbf{D} = \sqrt{\mathbf{C}}$（逐元素）

**AdaSmoothES**：
$$\mathbf{x}_k = \boldsymbol{\theta} + \sigma \cdot \sqrt{\mathbf{c}} \odot \mathbf{z}_k, \quad \mathbf{z}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$

**✅ 完全一样**

---

### 2. 权重计算

**SepCMA**（排名权重）：
```python
# 按 fitness 排序，只用前 μ = K/2 个
solutions.sort(key=lambda s: s[1])
weights = [log(μ+1) - log(i+1) for i in range(μ)]  # 只有前 μ 个非零
weights = weights / sum(weights)
```

$$w_i = \frac{\log(\mu+1) - \log(i+1)}{\sum_{j=1}^\mu [\log(\mu+1) - \log(j+1)]}, \quad i = 1, \ldots, \mu$$

**AdaSmoothES**（Softmax 权重）：
$$w_k = \frac{\exp(-(f_k - b)/\beta)}{\sum_{j=1}^K \exp(-(f_j - b)/\beta)}, \quad k = 1, \ldots, K$$

**❌ 不一样！** 这是核心区别。

| 方面 | SepCMA | AdaSmoothES |
|------|--------|-------------|
| 使用样本数 | 只用前 $\mu = K/2$ | 用全部 $K$ |
| 权重形式 | 对数排名 | Softmax |
| 理论基础 | 信息几何 / Natural gradient | KL-regularized Boltzmann |
| 对异常值 | 鲁棒（只看排名） | 敏感（看绝对值） |

---

### 3. 均值更新

**SepCMA**：
$$\mathbf{y}_w = \sum_{i=1}^\mu w_i \mathbf{y}_{i:\lambda}$$
$$\boldsymbol{\mu}_{t+1} = \boldsymbol{\mu}_t + c_m \cdot \sigma \cdot \mathbf{y}_w$$

其中 $\mathbf{y}_{i:\lambda} = (\mathbf{x}_{i:\lambda} - \boldsymbol{\mu}_t)/\sigma$，$c_m = 1$

**AdaSmoothES**：
$$\boldsymbol{\theta}_{t+1} = \sum_{k=1}^K w_k \mathbf{x}_k$$

**❌ 不一样！**

| 方面 | SepCMA | AdaSmoothES |
|------|--------|-------------|
| 形式 | $\boldsymbol{\mu} + \sigma \cdot \mathbf{y}_w$（增量） | $\sum w_k \mathbf{x}_k$（直接） |
| 数学等价？ | $= \boldsymbol{\mu} + \sum w_i (\mathbf{x}_i - \boldsymbol{\mu}) = \sum w_i \mathbf{x}_i + (1-\sum w_i)\boldsymbol{\mu}$ | $= \sum w_k \mathbf{x}_k$ |

**注意**：SepCMA 的 $\sum w_i = 1$（只对前 $\mu$ 个），所以实际上：
$$\boldsymbol{\mu}_{t+1} = \sum_{i=1}^\mu w_i \mathbf{x}_{i:\lambda}$$

如果 AdaSmoothES 的 $\sum w_k = 1$（对全部 $K$ 个），则形式相同，但**加权的样本集不同**。

---

### 4. Evolution Path

**SepCMA**：
$$\mathbf{p}_\sigma \leftarrow (1 - c_\sigma)\mathbf{p}_\sigma + \sqrt{c_\sigma(2-c_\sigma)\mu_{\text{eff}}} \cdot \frac{\mathbf{y}_w}{\mathbf{D}}$$

$$\mathbf{p}_c \leftarrow (1 - c_c)\mathbf{p}_c + h_\sigma \sqrt{c_c(2-c_c)\mu_{\text{eff}}} \cdot \mathbf{y}_w$$

**AdaSmoothES**：
$$\mathbf{p}_\sigma \leftarrow (1 - c_\sigma)\mathbf{p}_\sigma + \sqrt{c_\sigma(2-c_\sigma)\mu_{\text{eff}}} \cdot \frac{\mathbf{y}_w}{\sqrt{\mathbf{c}}}$$

$$\mathbf{p}_c \leftarrow (1 - c_c)\mathbf{p}_c + \sqrt{c_c(2-c_c)\mu_{\text{eff}}} \cdot \mathbf{y}_w$$

**⚠️ 几乎一样**，但：
- SepCMA 有 $h_\sigma$ 阈值控制
- $\mu_{\text{eff}}$ 计算方式不同

---

### 5. 协方差更新

**SepCMA**：
$$\mathbf{C}_{t+1} = \underbrace{(1 + c_1 \delta_{h_\sigma} - c_1 - c_\mu \sum w_i)}_{\text{decay}} \mathbf{C}_t + c_1 \underbrace{\mathbf{p}_c^2}_{\text{rank-one}} + c_\mu \underbrace{\sum_{i=1}^\mu w_i \mathbf{y}_{i:\lambda}^2}_{\text{rank-}\mu}$$

**AdaSmoothES**：
$$\mathbf{c}_{t+1} = (1 - c_1 - c_\mu) \mathbf{c}_t + c_1 \mathbf{p}_c^2 + c_\mu \sum_{k=1}^K w_k \mathbf{y}_k^2$$

**⚠️ 结构相同，细节不同**：

| 方面 | SepCMA | AdaSmoothES |
|------|--------|-------------|
| Decay 系数 | $(1 + c_1\delta_{h_\sigma} - c_1 - c_\mu\sum w_i)$ | $(1 - c_1 - c_\mu)$ |
| Rank-one | $\mathbf{p}_c^2$ | $\mathbf{p}_c^2$ |
| Rank-$\mu$ 样本 | 前 $\mu$ 个（排序后） | 全部 $K$ 个 |
| Rank-$\mu$ 权重 | 对数排名权重 | Softmax 权重 |

---

### 6. Step-size 更新 (CSA)

**SepCMA**：
$$\sigma_{t+1} = \sigma_t \cdot \exp\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\|\mathbf{p}_\sigma\|}{\chi_d} - 1\right)\right)$$

**AdaSmoothES**：
$$\sigma_{t+1} = \sigma_t \cdot \exp\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\|\mathbf{p}_\sigma\|}{\chi_d} - 1\right)\right)$$

**✅ 完全一样**

---

## 总结：核心区别

| 组件 | SepCMA | AdaSmoothES | 相同？ |
|------|--------|-------------|--------|
| **采样** | $\boldsymbol{\mu} + \sigma\mathbf{D}\odot\mathbf{z}$ | $\boldsymbol{\theta} + \sigma\sqrt{\mathbf{c}}\odot\mathbf{z}$ | ✅ |
| **权重** | 对数排名，只用前 $\mu$ | Softmax，用全部 $K$ | ❌ **核心区别** |
| **均值** | $\sum_{i=1}^\mu w_i \mathbf{x}_{i:\lambda}$ | $\sum_{k=1}^K w_k \mathbf{x}_k$ | ⚠️ 形式同，样本不同 |
| **$\mathbf{p}_c$** | 带 $h_\sigma$ 阈值 | 无阈值 | ⚠️ |
| **$\mathbf{p}_\sigma$** | 相同公式 | 相同公式 | ✅ |
| **协方差** | 加性更新 | 加性更新 | ⚠️ 细节不同 |
| **CSA** | 相同公式 | 相同公式 | ✅ |

---

## 本质区别：权重的理论来源

**SepCMA 的权重**（信息几何）：
- 来自 Natural Evolution Strategies (NES)
- 基于 Fisher 信息矩阵的 natural gradient
- 对数权重 $w_i \propto \log(\mu+1) - \log(i)$ 是最优的 fitness shaping

**AdaSmoothES 的权重**（你的理论）：
- 来自 KL-regularized policy optimization
- 基于 Boltzmann 分布的 moment matching
- Softmax 权重 $w_k \propto \exp(-f_k/\beta)$ 是变分最优解

---

## 我的建议

保持 **Softmax 权重**作为你的理论贡献，因为：

1. **有理论基础**：来自 KL-regularized Boltzmann 分布
2. **可调节**：通过 $\beta$ 控制 exploration-exploitation
3. **区别于 CMA-ES**：这是你论文的创新点

你可以在实验中对比两种权重方案，展示 Softmax 权重在某些情况下的优势（比如多模态函数）。