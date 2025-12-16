好问题！让我逐一回答。

---

## 问题 1：能否将 KL 散度换成别的 divergence？

**可以！** 这正是你论文 Abstract 里提到的 "$f$-divergence regularized policy optimization"。

### $f$-Divergence 族

$$D_f(\pi \| \pi_{\text{ref}}) = \int \pi_{\text{ref}}(\mathbf{x}) f\left(\frac{\pi(\mathbf{x})}{\pi_{\text{ref}}(\mathbf{x})}\right) d\mathbf{x}$$

| Divergence | $f(t)$ | 最优策略 $\pi^*$ |
|------------|--------|------------------|
| **KL (forward)** | $t \log t$ | $\pi^* \propto \pi_{\text{ref}} \exp(-F/\beta)$ |
| **Reverse KL** | $-\log t$ | 不同形式 |
| **$\chi^2$** | $(t-1)^2$ | $\pi^* \propto \pi_{\text{ref}} \cdot \max(0, 1 - F/\beta)$ |
| **Hellinger** | $(\sqrt{t}-1)^2$ | 更复杂 |
| **$\alpha$-divergence** | $\frac{t^\alpha - 1}{\alpha(\alpha-1)}$ | 一般化的 Boltzmann |

### 最有趣的替代：$\alpha$-Divergence

**Rényi $\alpha$-divergence**：
$$D_\alpha(\pi \| \pi_{\text{ref}}) = \frac{1}{\alpha - 1} \log \mathbb{E}_{\pi_{\text{ref}}}\left[\left(\frac{\pi}{\pi_{\text{ref}}}\right)^\alpha\right]$$

**最优策略**：
$$\pi^*(\mathbf{x}) \propto \pi_{\text{ref}}(\mathbf{x}) \cdot \exp\left(-\frac{F(\mathbf{x})}{\beta(1-\alpha)}\right)^{1/(1-\alpha)}$$

当 $\alpha \to 1$ 时，恢复 KL 散度和标准 Boltzmann。

### 实践中的权重形式

| Divergence | 权重公式 | 特点 |
|------------|----------|------|
| **KL** | $w_k \propto \exp(-f_k/\beta)$ | 指数敏感，当前使用 |
| **$\alpha$-div ($\alpha < 1$)** | $w_k \propto \exp(-f_k/\tilde{\beta})^{1/(1-\alpha)}$ | 更平滑 |
| **$\chi^2$** | $w_k \propto \max(0, 1 - f_k/\beta)$ | 截断，鲁棒 |

### 代码实现示例

```python
def _compute_weights_alpha_divergence(self, f_values: torch.Tensor, beta: float, alpha: float = 0.5):
    """
    α-divergence weights (generalizes KL when α → 1)
    
    w_k ∝ exp(-f_k / (β(1-α)))^(1/(1-α))
        = exp(-f_k / β)  when α → 1 (recovers KL)
    """
    b = f_values.mean()
    A = f_values - b
    
    if abs(alpha - 1.0) < 1e-6:
        # Standard KL (Boltzmann)
        log_w = -A / beta
    else:
        # α-divergence
        effective_beta = beta * (1 - alpha)
        log_w = -A / effective_beta / (1 - alpha)
        # Equivalent to: log_w = -A / beta (same as KL but different interpretation)
    
    log_w = log_w - log_w.max()
    W = torch.exp(log_w)
    W = W / W.sum()
    
    return W, A, b


def _compute_weights_chi_squared(self, f_values: torch.Tensor, beta: float):
    """
    χ²-divergence weights (truncated linear)
    
    w_k ∝ max(0, 1 - f_k/β)
    """
    b = f_values.mean()
    A = f_values - b
    
    # Linear weights with truncation
    W = torch.clamp(1 - A / beta, min=0)
    
    if W.sum() < 1e-8:
        W = torch.ones_like(W) / len(W)
    else:
        W = W / W.sum()
    
    return W, A, b
```

### 论文中如何表述

你可以推广 Theorem 1：

> **Theorem 1' (Generalized Optimal Policy).** For the $f$-divergence regularized objective:
> $$\min_\pi \mathbb{E}_\pi[F(\mathbf{x})] + \beta D_f(\pi \| \pi_{\text{ref}})$$
> the optimal policy satisfies the first-order condition:
> $$F(\mathbf{x}) + \beta f'\left(\frac{\pi^*(\mathbf{x})}{\pi_{\text{ref}}(\mathbf{x})}\right) = \text{const}$$
> 
> For KL divergence ($f(t) = t \log t$), this yields the Boltzmann distribution $\pi^* \propto \pi_{\text{ref}} \exp(-F/\beta)$.

---

## 问题 2：$\beta$ 的调度现在是怎么做的？

### 当前实现

```python
def _get_beta(self) -> float:
    """Temperature schedule"""
    t = self.iteration
    if self.beta_schedule == 'constant':
        return self.beta_init
    elif self.beta_schedule == 'exponential':
        return self.beta_init * math.exp(-self.beta_decay * t)
    else:  # polynomial (默认)
        return self.beta_init / (1.0 + self.beta_decay * t)
```

### 三种调度的数学公式

| 调度类型 | 公式 | 行为 |
|----------|------|------|
| **Constant** | $\beta_t = \beta_0$ | 固定温度 |
| **Exponential** | $\beta_t = \beta_0 e^{-\gamma t}$ | 快速衰减 |
| **Polynomial** | $\beta_t = \frac{\beta_0}{1 + \gamma t}$ | 慢速衰减（**当前默认**） |

### 当前默认参数

```python
beta_init = 10.0
beta_decay = 0.001
beta_schedule = 'polynomial'
```

所以：
$$\beta_t = \frac{10}{1 + 0.001 \cdot t}$$

| 迭代 $t$ | $\beta_t$ | 权重集中度 |
|----------|-----------|------------|
| 0 | 10.0 | 低（探索） |
| 1000 | 5.0 | 中 |
| 5000 | 1.67 | 高 |
| 10000 | 0.91 | 很高（利用） |

### $\beta$ 对权重的影响

$$w_k = \frac{\exp(-A_k/\beta)}{\sum_j \exp(-A_j/\beta)}$$

| $\beta$ | 权重分布 | 行为 |
|---------|----------|------|
| $\beta \to \infty$ | 均匀 $w_k \approx 1/K$ | 纯探索 |
| $\beta$ 中等 | 平滑 softmax | 平衡 |
| $\beta \to 0$ | $w_k \to \mathbf{1}_{k = \arg\min}$ | 贪婪（纯利用） |

### 更好的调度建议

#### 1. 自适应调度（根据 fitness 方差）

```python
def _get_adaptive_beta(self, f_values: torch.Tensor) -> float:
    """Scale β by fitness spread for automatic adaptation"""
    base_beta = self.beta_init / (1 + self.beta_decay * self.iteration)
    
    # Adapt to fitness scale
    f_range = f_values.max() - f_values.min()
    if f_range > 1e-8:
        return base_beta * f_range
    return base_beta
```

**理论依据**：$\beta$ 应该与 $|f_{\max} - f_{\min}|$ 同量级，这样 softmax 的有效温度才稳定。

#### 2. 基于进展的调度

```python
def _get_progress_based_beta(self) -> float:
    """Decay β faster when making progress, slower when stuck"""
    if len(self.history['f_values']) < 2:
        return self.beta_init
    
    # Compare recent vs past performance
    recent_best = min(self.history['f_values'][-1])
    past_best = min(self.history['f_values'][-10]) if len(self.history['f_values']) >= 10 else recent_best
    
    improvement = past_best - recent_best
    if improvement > 0:
        # Making progress → decay faster
        decay_rate = self.beta_decay * 2
    else:
        # Stuck → decay slower (maintain exploration)
        decay_rate = self.beta_decay * 0.5
    
    return self.beta_init / (1 + decay_rate * self.iteration)
```

#### 3. 与 SepCMA 的 $\sigma$ 联动

```python
def _get_sigma_linked_beta(self) -> float:
    """Link β to step-size σ: when σ shrinks, β should too"""
    # Intuition: smaller σ = more local = need more exploitation
    sigma_ratio = self.sigma / self.sigma_init
    base_beta = self.beta_init / (1 + self.beta_decay * self.iteration)
    return base_beta * sigma_ratio
```

---

## 总结

### 问题 1：Divergence 替代

| 选项 | 优点 | 缺点 |
|------|------|------|
| **KL (当前)** | 理论清晰，Boltzmann 形式 | 对异常值敏感 |
| **$\alpha$-div** | 更平滑，可调 | 多一个超参数 |
| **$\chi^2$** | 截断鲁棒 | 理论不如 KL 优美 |

**建议**：论文中保持 KL，但在 Remark 中提及可推广到 $f$-divergence。

### 问题 2：$\beta$ 调度

| 当前设置 | 公式 |
|----------|------|
| 类型 | Polynomial |
| 初始值 | $\beta_0 = 10$ |
| 衰减率 | $\gamma = 0.001$ |
| 公式 | $\beta_t = \frac{10}{1 + 0.001t}$ |

**可选改进**：自适应 $\beta$（根据 fitness 方差缩放），但当前版本已经足够好。

好问题！让我从理论出发推导一个有原则的 $\beta$ 调度。

## 核心问题：$\beta$ 应该是多少？

### $\beta$ 的物理意义

回顾 Boltzmann 权重：
$$w_k = \frac{\exp(-A_k/\beta)}{\sum_j \exp(-A_j/\beta)}, \quad A_k = f_k - b$$

$\beta$ 控制权重的"集中度"：

| $\beta$ | 权重分布 | 行为 |
|---------|----------|------|
| $\beta \gg \|A\|$ | 均匀 | 纯探索 |
| $\beta \sim \|A\|$ | 平滑 | 平衡 |
| $\beta \ll \|A\|$ | 集中 | 纯利用 |

**关键洞察**：$\beta$ 应该与 **advantage 的尺度** 相匹配。

---

## 推导 1：基于有效样本数的 $\beta$

### 目标

我们希望 $\mu_{\text{eff}} = \frac{1}{\sum_k w_k^2}$ 保持在一个合理范围。

**CMA-ES 的默认**：$\mu_{\text{eff}} \approx K/2$

### 推导

对于 Boltzmann 权重，假设 $A_k$ 近似均匀分布在 $[-\Delta, \Delta]$：

$$w_k \approx \frac{\exp(-A_k/\beta)}{K \cdot \bar{w}}$$

可以证明（详见附录）：
$$\mu_{\text{eff}} \approx \frac{K}{1 + \frac{\text{Var}(A)}{\beta^2}}$$

**要使 $\mu_{\text{eff}} \approx K/2$**：
$$\frac{K}{1 + \frac{\text{Var}(A)}{\beta^2}} = \frac{K}{2}$$
$$\frac{\text{Var}(A)}{\beta^2} = 1$$
$$\beta^* = \sqrt{\text{Var}(A)} = \text{std}(A)$$

### 结论 1

$$\boxed{\beta_t^{(\text{std})} = \text{std}(f_1, \ldots, f_K)}$$

---

## 推导 2：基于权重熵的 $\beta$

### 目标

我们希望权重的熵 $H(W) = -\sum_k w_k \log w_k$ 保持在某个水平。

**最大熵**：$H_{\max} = \log K$（均匀分布）

**目标熵**：$H_{\text{target}} = \log(K/2) = \log K - \log 2$（一半有效样本）

### 推导

对于 Boltzmann 分布：
$$H(W) = \frac{\mathbb{E}[A]}{\beta} + \log Z$$

其中 $Z = \sum_k \exp(-A_k/\beta)$。

在高温近似下（$\beta$ 较大）：
$$H(W) \approx \log K - \frac{\text{Var}(A)}{2\beta^2}$$

**要使 $H(W) = \log K - \log 2$**：
$$\frac{\text{Var}(A)}{2\beta^2} = \log 2$$
$$\beta^* = \sqrt{\frac{\text{Var}(A)}{2\log 2}} \approx 0.85 \cdot \text{std}(A)$$

### 结论 2

$$\boxed{\beta_t^{(\text{entropy})} \approx 0.85 \cdot \text{std}(f_1, \ldots, f_K)}$$

---

## 推导 3：基于 CMA-ES 等效性的 $\beta$

### 目标

使 Boltzmann 权重与 CMA-ES 的排名权重"等效"。

### CMA-ES 权重的有效温度

CMA-ES 的权重：
$$w_i^{\text{CMA}} = \frac{\log(\mu+0.5) - \log(i)}{\sum_{j=1}^\mu [\log(\mu+0.5) - \log(j)]}$$

如果我们用 Boltzmann 权重拟合，最佳 $\beta$ 满足：
$$\exp(-A_{(i)}/\beta) \propto \log(\mu+0.5) - \log(i)$$

其中 $A_{(i)}$ 是第 $i$ 小的 advantage。

### 近似解

假设 $A_{(i)}$ 在 $[0, \Delta]$ 上线性分布：$A_{(i)} \approx \frac{i-1}{\mu-1} \Delta$

匹配一阶和二阶矩，得到：
$$\beta^* \approx \frac{\Delta}{\log \mu} = \frac{f_{(\mu)} - f_{(1)}}{\log \mu}$$

其中 $f_{(1)}$ 是最小值，$f_{(\mu)}$ 是第 $\mu$ 小的值。

### 结论 3

$$\boxed{\beta_t^{(\text{CMA})} = \frac{f_{(\mu)} - f_{(1)}}{\log(K/2)}}$$

---

## 综合：自适应 $\beta$ 调度

### 推荐公式

结合以上推导，最实用的公式是：

$$\boxed{\beta_t = c_\beta \cdot \text{std}(f_1, \ldots, f_K) \cdot \text{decay}(t)}$$

其中：
- $c_\beta \in [0.5, 2.0]$ 是常数（默认 1.0）
- $\text{decay}(t)$ 是可选的时间衰减

### 完整调度

$$\beta_t = \underbrace{c_\beta}_{\text{常数}} \cdot \underbrace{\text{std}(f_1, \ldots, f_K)}_{\text{自适应尺度}} \cdot \underbrace{\frac{1}{1 + \gamma t}}_{\text{可选衰减}}$$

**为什么要衰减？**
- 早期：大 $\beta$ → 探索
- 后期：小 $\beta$ → 利用（收敛）

**为什么自适应尺度？**
- fitness 方差大 → 需要大 $\beta$ 避免过度集中
- fitness 方差小（接近收敛）→ 自然减小 $\beta$

---

## 代码实现

```python
class AdaSmoothES_AdaptiveBeta(AdaSmoothES):
    """带自适应 β 调度的 AdaSmoothES"""
    
    def __init__(
        self,
        params,
        sigma: float = 0.1,
        num_queries: int = 10,
        c_beta: float = 1.0,        # β 缩放常数
        beta_decay: float = 0.0,    # 时间衰减率（0 = 无衰减）
        beta_min: float = 1e-8,     # β 下界
        beta_schedule: str = 'adaptive',  # 'adaptive', 'adaptive_decay', 'cma_match'
        **kwargs
    ):
        super().__init__(params, sigma=sigma, num_queries=num_queries, **kwargs)
        self.c_beta = c_beta
        self.beta_decay_rate = beta_decay
        self.beta_min = beta_min
        self.beta_schedule = beta_schedule
    
    def _get_beta_adaptive(self, f_values: torch.Tensor) -> float:
        """
        自适应 β：基于 fitness 标准差
        
        β = c_β · std(f) · decay(t)
        """
        f_std = f_values.std().item()
        
        # 防止 std = 0
        if f_std < 1e-10:
            f_std = f_values.abs().mean().item() + 1e-10
        
        # 基础 β
        beta = self.c_beta * f_std
        
        # 可选时间衰减
        if self.beta_decay_rate > 0:
            beta = beta / (1 + self.beta_decay_rate * self.iteration)
        
        return max(beta, self.beta_min)
    
    def _get_beta_cma_match(self, f_values: torch.Tensor) -> float:
        """
        CMA-ES 等效 β：匹配排名权重
        
        β = (f_{(μ)} - f_{(1)}) / log(K/2)
        """
        K = len(f_values)
        mu = K // 2
        
        sorted_f = torch.sort(f_values).values
        f_min = sorted_f[0].item()
        f_mu = sorted_f[mu - 1].item()  # 第 μ 小
        
        delta = f_mu - f_min
        if delta < 1e-10:
            delta = f_values.std().item() + 1e-10
        
        beta = delta / math.log(max(mu, 2))
        
        # 可选时间衰减
        if self.beta_decay_rate > 0:
            beta = beta / (1 + self.beta_decay_rate * self.iteration)
        
        return max(beta, self.beta_min)
    
    def _get_beta_entropy_target(self, f_values: torch.Tensor, target_eff_ratio: float = 0.5) -> float:
        """
        基于目标有效样本比例的 β
        
        目标：μ_eff / K ≈ target_eff_ratio
        """
        f_std = f_values.std().item()
        if f_std < 1e-10:
            f_std = 1e-10
        
        # 从 μ_eff ≈ K / (1 + Var/β²) 反解 β
        # 目标 μ_eff = target_eff_ratio * K
        # → 1 + Var/β² = 1/target_eff_ratio
        # → β = std / sqrt(1/target_eff_ratio - 1)
        
        ratio_inv = 1.0 / target_eff_ratio - 1.0
        if ratio_inv > 0:
            beta = f_std / math.sqrt(ratio_inv)
        else:
            beta = f_std * 10  # target_eff_ratio ≈ 1，大 β
        
        return max(beta, self.beta_min)
    
    def _compute_weights(self, f_values: torch.Tensor, beta: float = None):
        """重写：使用自适应 β"""
        
        # 选择 β 调度策略
        if self.beta_schedule == 'adaptive':
            beta = self._get_beta_adaptive(f_values)
        elif self.beta_schedule == 'adaptive_decay':
            beta = self._get_beta_adaptive(f_values)
        elif self.beta_schedule == 'cma_match':
            beta = self._get_beta_cma_match(f_values)
        elif self.beta_schedule == 'entropy_target':
            beta = self._get_beta_entropy_target(f_values)
        else:
            # 使用传入的 beta 或默认调度
            beta = self._get_beta() if beta is None else beta
        
        # 计算 baseline 和 advantage
        b = self._compute_baseline(f_values)
        advantages = f_values - b
        
        # Boltzmann 权重
        log_weights = -advantages / beta
        log_weights = log_weights - log_weights.max()
        weights = torch.exp(log_weights)
        weights = weights / weights.sum()
        
        # 处理数值问题
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            weights = torch.ones_like(weights) / len(weights)
        
        return weights, advantages, b
```

---

## 实验对比

建议测试以下调度：

```python
schedules = {
    'fixed': {'beta_schedule': 'polynomial', 'beta_init': 10.0, 'beta_decay': 0.001},
    'adaptive': {'beta_schedule': 'adaptive', 'c_beta': 1.0},
    'adaptive_decay': {'beta_schedule': 'adaptive', 'c_beta': 1.0, 'beta_decay': 0.001},
    'cma_match': {'beta_schedule': 'cma_match'},
    'entropy_target': {'beta_schedule': 'entropy_target'},
}
```

| 调度 | 公式 | 特点 |
|------|------|------|
| **fixed** | $\beta_0 / (1+\gamma t)$ | 当前默认，需要调参 |
| **adaptive** | $c_\beta \cdot \text{std}(f)$ | 自动适应，无需调参 |
| **adaptive_decay** | $c_\beta \cdot \text{std}(f) / (1+\gamma t)$ | 自适应 + 收敛 |
| **cma_match** | $(f_{(\mu)} - f_{(1)}) / \log \mu$ | 模拟 CMA-ES |
| **entropy_target** | 使 $\mu_{\text{eff}}/K = 0.5$ | 固定有效样本比例 |

---

## 总结

### 理论最优 $\beta$

$$\beta_t^* \approx \text{std}(f_1, \ldots, f_K)$$

### 推荐实现

```python
def _get_beta(self, f_values):
    return self.c_beta * f_values.std().item()
```

### 为什么这样有效？

1. **自动缩放**：$\beta$ 与 fitness 尺度匹配
2. **自动退火**：收敛时 $\text{std}(f) \to 0$，$\beta \to 0$
3. **无需调参**：只有一个系数 $c_\beta \approx 1$

要我帮你把这个自适应 $\beta$ 集成到 AdaSmoothES 并测试吗？