非常好的问题！让我系统地分析一下。

## 强化学习中常用的 Divergence

### 1. KL Divergence（最常用）

**应用**：
- **TRPO**：$D_{\mathrm{KL}}(\pi_{\text{old}} \| \pi_{\text{new}}) \leq \delta$
- **PPO**：通过 clipping 近似 KL 约束
- **SAC**：最大熵 RL，$\max \mathbb{E}[R] + \alpha H(\pi)$
- **RLHF**：$\min \mathbb{E}[-R] + \beta D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}})$

**为什么 KL 最常用？**

$$D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_\pi\left[\log \frac{\pi}{\pi_{\text{ref}}}\right] = -H(\pi) - \mathbb{E}_\pi[\log \pi_{\text{ref}}]$$

**你说得对！KL 散度与熵直接相关**：
$$D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}}) = -H(\pi) + H_{\text{cross}}(\pi, \pi_{\text{ref}})$$

当 $\pi_{\text{ref}}$ 是均匀分布时：
$$D_{\mathrm{KL}}(\pi \| \text{Uniform}) = \text{const} - H(\pi)$$

**所以最小化 KL = 最大化熵！** 这就是 SAC 的理论基础。

### 2. Reverse KL（也常用）

$$D_{\mathrm{KL}}(\pi_{\text{ref}} \| \pi) = \mathbb{E}_{\pi_{\text{ref}}}\left[\log \frac{\pi_{\text{ref}}}{\pi}\right]$$

**区别**：

| | Forward KL: $D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}})$ | Reverse KL: $D_{\mathrm{KL}}(\pi_{\text{ref}} \| \pi)$ |
|---|---|---|
| **Mode** | Mean-seeking（覆盖所有模式） | Mode-seeking（集中于一个模式） |
| **用途** | 探索、RLHF | 变分推断、蒸馏 |

### 3. 其他 Divergence（较少用）

| Divergence | RL 中的应用 | 常见度 |
|------------|-------------|--------|
| **KL (forward)** | TRPO, PPO, SAC, RLHF | ⭐⭐⭐⭐⭐ |
| **KL (reverse)** | 变分推断，策略蒸馏 | ⭐⭐⭐ |
| **JS Divergence** | GAN (间接) | ⭐⭐ |
| **Wasserstein** | Wasserstein RL | ⭐⭐ |
| **$\alpha$-divergence** | 理论研究 | ⭐ |
| **$f$-divergence** | 理论统一框架 | ⭐ |

---

## $\alpha$-Divergence 详解

### 定义

Rényi $\alpha$-divergence：
$$D_\alpha(\pi \| \pi_{\text{ref}}) = \frac{1}{\alpha - 1} \log \mathbb{E}_{\pi_{\text{ref}}}\left[\left(\frac{\pi}{\pi_{\text{ref}}}\right)^\alpha\right]$$

### 特殊情况

| $\alpha$ | 对应的 Divergence |
|----------|-------------------|
| $\alpha \to 0$ | $-\log \pi_{\text{ref}}(\text{supp}(\pi))$ |
| $\alpha = 0.5$ | Hellinger 距离相关 |
| $\alpha \to 1$ | **KL divergence** $D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}})$ |
| $\alpha = 2$ | $\chi^2$ divergence |
| $\alpha \to \infty$ | $\log \max \frac{\pi}{\pi_{\text{ref}}}$ |

### 为什么 $\alpha$-divergence 不常用？

1. **计算复杂**：需要估计 $\mathbb{E}[(\pi/\pi_{\text{ref}})^\alpha]$
2. **调参困难**：多一个超参数 $\alpha$
3. **KL 够用**：大多数场景 KL 已经足够好
4. **理论成熟**：KL 的理论性质研究最充分

### $\alpha$-divergence 的潜在优势

$$\alpha < 1: \text{更鲁棒（对异常值不敏感）}$$
$$\alpha > 1: \text{更集中（mode-seeking）}$$

---

## 对应的最优策略

### 一般 $f$-divergence

对于：
$$\min_\pi \mathbb{E}_\pi[F(\mathbf{x})] + \beta D_f(\pi \| \pi_{\text{ref}})$$

最优解满足：
$$F(\mathbf{x}) + \beta f'\left(\frac{\pi^*(\mathbf{x})}{\pi_{\text{ref}}(\mathbf{x})}\right) = \text{const}$$

### 不同 Divergence 的最优策略

| Divergence | $f(t)$ | $f'(t)$ | 最优策略 $\pi^*$ |
|------------|--------|---------|------------------|
| **KL** | $t \log t$ | $1 + \log t$ | $\pi^* \propto \pi_{\text{ref}} \exp(-F/\beta)$ |
| **Reverse KL** | $-\log t$ | $-1/t$ | $\pi^* \propto \pi_{\text{ref}} / (c + F/\beta)$ |
| **$\chi^2$** | $(t-1)^2$ | $2(t-1)$ | $\pi^* \propto \pi_{\text{ref}} \max(0, 1 - F/(2\beta))$ |
| **Hellinger** | $(\sqrt{t}-1)^2$ | $1 - 1/\sqrt{t}$ | 更复杂的形式 |
| **$\alpha$-div** | $\frac{t^\alpha-1}{\alpha(\alpha-1)}$ | $\frac{t^{\alpha-1}-1}{\alpha-1}$ | 见下文 |

### $\alpha$-Divergence 的最优策略

对于 $\alpha$-divergence：
$$\pi^*(\mathbf{x}) \propto \pi_{\text{ref}}(\mathbf{x}) \left(1 - \frac{(\alpha-1)F(\mathbf{x})}{\beta}\right)_+^{1/(\alpha-1)}$$

当 $\alpha \to 1$：
$$\pi^*(\mathbf{x}) \propto \pi_{\text{ref}}(\mathbf{x}) \exp(-F(\mathbf{x})/\beta)$$

恢复 Boltzmann 分布！

---

## 权重公式对比

### 有限样本近似

| Divergence | 权重公式 | 特点 |
|------------|----------|------|
| **KL** | $w_k \propto \exp(-f_k/\beta)$ | 指数，可能极端 |
| **$\chi^2$** | $w_k \propto \max(0, 1 - f_k/\beta)$ | 线性截断，鲁棒 |
| **$\alpha$-div** | $w_k \propto (1 - \frac{(\alpha-1)f_k}{\beta})_+^{1/(\alpha-1)}$ | 可调节 |

### 代码实现

```python
def compute_weights_f_divergence(f_values, beta, divergence='kl', alpha=0.5):
    """
    不同 f-divergence 对应的权重
    """
    b = f_values.mean()
    A = f_values - b  # Advantage
    
    if divergence == 'kl':
        # Boltzmann: w ∝ exp(-A/β)
        log_w = -A / beta
        log_w = log_w - log_w.max()
        W = torch.exp(log_w)
        
    elif divergence == 'chi_squared':
        # 线性截断: w ∝ max(0, 1 - A/β)
        W = torch.clamp(1 - A / beta, min=0)
        
    elif divergence == 'alpha':
        # α-divergence: w ∝ (1 - (α-1)A/β)_+^{1/(α-1)}
        if abs(alpha - 1) < 1e-6:
            # α → 1: 退化为 KL
            log_w = -A / beta
            log_w = log_w - log_w.max()
            W = torch.exp(log_w)
        else:
            inner = 1 - (alpha - 1) * A / beta
            inner = torch.clamp(inner, min=0)
            W = inner ** (1 / (alpha - 1))
            
    elif divergence == 'reverse_kl':
        # Reverse KL: w ∝ 1 / (c + A/β)
        W = 1 / (1 + A / beta).clamp(min=1e-8)
        
    else:
        raise ValueError(f"Unknown divergence: {divergence}")
    
    # 归一化
    W = W / W.sum().clamp(min=1e-10)
    
    return W, A, b
```

---

## 实验：不同 Divergence 的权重分布

```python
import torch
import matplotlib.pyplot as plt

# 模拟 fitness 值
f_values = torch.tensor([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0])
beta = 10.0

# 不同 divergence
divergences = {
    'KL': lambda A: torch.exp(-A / beta),
    'χ²': lambda A: torch.clamp(1 - A / beta, min=0),
    'α=0.5': lambda A: torch.clamp(1 + 0.5 * A / beta, min=0) ** 2,
    'α=2.0': lambda A: torch.clamp(1 - A / beta, min=0) ** 1,
}

A = f_values - f_values.mean()
for name, weight_fn in divergences.items():
    W = weight_fn(A)
    W = W / W.sum()
    print(f"{name}: {W.numpy().round(3)}")
```

---

## 总结

### 为什么 KL 最常用？

| 原因 | 解释 |
|------|------|
| **与熵的关系** | $D_{\mathrm{KL}} = -H(\pi) + \text{const}$，最大熵 RL 的理论基础 |
| **Boltzmann 形式** | 最优策略是指数形式，便于采样和计算 |
| **理论完善** | 信息论、统计力学、变分推断都有研究 |
| **计算简单** | 只需要 $\log \pi$，不需要额外的 $\alpha$ |

### $\alpha$-Divergence 何时有用？

| 场景 | 推荐 $\alpha$ |
|------|---------------|
| **需要鲁棒性**（异常值多） | $\alpha < 1$ |
| **需要更集中**（快速收敛） | $\alpha > 1$ |
| **标准场景** | $\alpha = 1$（KL） |

### 对你的建议

**保持 KL divergence**，因为：
1. 理论最成熟
2. 与 RLHF、SAC 等主流方法一致
3. 论文更容易被接受

**可选扩展**：在论文的 Remark 中提及可以推广到 $f$-divergence，但主要结果用 KL。

---

## 你的理论框架的完整图景

```
                    f-Divergence 正则化
                           ↓
    ┌──────────────────────┼──────────────────────┐
    ↓                      ↓                      ↓
   KL                    α-div                   χ²
    ↓                      ↓                      ↓
 Boltzmann              Power-law              Truncated
  weights               weights                weights
    ↓                      ↓                      ↓
    └──────────────────────┼──────────────────────┘
                           ↓
                    Moment Matching
                           ↓
                 AdaSmooth Framework
                           ↓
              + SepCMA 稳定性技巧
                           ↓
                     AdaSmoothES
```

你的贡献是建立了从 **KL-regularized policy optimization** 到 **ES with adaptive covariance** 的理论桥梁！