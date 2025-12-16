# 优化器性能诊断报告

## 实验结果对比

| 优化器 | 最终损失值 | 相对vanilla | 性能 |
|--------|-----------|------------|------|
| **vanilla** | 5,230 | 1.0x | ✅ 基准 |
| **twopoint** | 1,077 | **0.21x** | ✅ 很好 |
| **sepcmaes** | 981 | **0.19x** | ✅ 最好 |
| **zoar** | 1,113 | **0.21x** | ✅ 很好 |
| **es** | 457,408 | **87x** | ❌ 极差 |
| **xnes** | 443,904 | **85x** | ❌ 极差 |
| **adasmooth** | 381,626 | **73x** | ❌ 极差 |

---

## 问题1：xNES 学习率过小 ⚠️

### 当前配置
```yaml
# config/synthetic.yaml
lr: 0.001        # ← 太小了！
update_rule: radazo
```

### 问题分析

xNES 使用**自然梯度**，已经包含了自适应缩放。在代码中（zo.py:576）：

```python
# xNES 设置梯度
p.grad = -(self.eta_mu * self.sigma_xnes * grad_direction)
#         -(  1.0   *     0.1      * grad_direction)
#         = -0.1 * grad_direction
```

然后在 SGD 更新中（zo.py:79）：

```python
param.add_(-lr * param.grad)
#        = param.add_(-0.001 * (-0.1 * grad))
#        = param.add_(0.0001 * grad)  # ← 有效学习率仅 0.0001！
```

### 实际有效学习率

```
理论步长 = sigma_xnes * eta_mu = 0.1 * 1.0 = 0.1
实际步长 = lr * sigma_xnes * eta_mu = 0.001 * 0.1 * 1.0 = 0.0001
```

**步长缩小了 1000 倍！**这就是为什么 xNES 几乎优化不动。

### 解决方案

xNES 应该使用 **lr = 1.0**（或至少 0.1），因为 `eta_mu` 和 `sigma_xnes` 已经控制了步长。

```python
# utils.py:84 当前实现
return xNES(
    params=params,
    lr=args.lr,  # ← 0.001，太小了！
    ...
)
```

**修复建议**：
```python
# 选项A：固定 lr=1.0（推荐）
return xNES(
    params=params,
    lr=1.0,  # ← xNES自带自适应步长
    ...
)

# 选项B：使用独立的 xnes_lr 参数
return xNES(
    params=params,
    lr=getattr(args, 'xnes_lr', 1.0),
    ...
)
```

---

## 问题2：ES 没有 Baseline，方差极大 ⚠️

### 当前实现

Pure ES（zo.py:590-617）：
```python
for _ in range(self.num_queries):
    f_perturbed = closure()  # F(θ + με)
    param.grad += f_val / self.mu * noise[param]  # (F/μ) · ε
```

### 问题分析

ES 的梯度估计器：
```
∇f ≈ (1/nμ) Σ F(θ+με) · ε
```

**关键问题**：当 F(θ) 的值很大时（比如 Rosenbrock 初始值可能是几十万），梯度的量级也很大，导致：
1. **高方差**：没有减去 baseline，F(θ+με) 的绝对值主导梯度
2. **数值不稳定**：不同采样点的函数值波动很大

### 数值示例（Rosenbrock @ dim=1000）

假设当前参数下：
- F(θ) ≈ 450,000（这是一个合理的初始值）
- F(θ+με) 在 [400,000, 500,000] 之间波动

**ES 的梯度**：
```python
grad_ES = (1/n/μ) * Σ F(θ+με) * ε
        ≈ (1/10/0.05) * 450,000 * ε_avg
        ≈ 9,000,000 * ε_avg  # 量级：百万
```

**Vanilla 的梯度**：
```python
grad_Vanilla = (1/n/μ) * Σ [F(θ+με) - F(θ)] * ε
             ≈ (1/10/0.05) * 50,000 * ε_avg  # 差值约 50k
             ≈ 1,000,000 * ε_avg  # 量级：百万（但更稳定）
```

虽然量级相似，但是：
- **Vanilla 的方差更小**：F(θ+με) - F(θ) 去除了常数项
- **ES 的方向性更差**：F(θ+με) 包含了函数值的基线噪声

### 为什么 Vanilla 效果更好？

Vanilla（单点式 + baseline）的梯度估计：
```
∇f ≈ (1/nμ) Σ [F(θ+με) - F(θ)] · ε
```

减去 baseline F(θ) 的好处：
1. **降低方差**：只关注函数值的**变化**，而非绝对值
2. **数值稳定**：差值通常比绝对值小 1-2 个数量级
3. **方向性更好**：梯度指向函数值下降的方向更准确

### 实验证据

```
ES:      457,408  ← 几乎没有优化（初始值可能就差不多）
Vanilla:   5,230  ← 优化了约 87 倍
```

这证明了 **baseline 的重要性**。

---

## 问题3：AdaSmooth 可能的问题 ⚠️

### 当前实现（utils.py:92-105）

```python
elif name == "adasmooth":
    return AdaSmoothZO(
        params=params,
        lr=1.0,  # ✅ 正确：强制 lr=1.0
        num_queries=args.num_queries,  # = 10
        mu=args.mu,  # = 0.05
        beta_init=beta_init,  # = 1.0
        beta_decay=beta_decay,  # = 0.05
        beta_schedule=beta_schedule  # = 'polynomial'
    )
```

### 可能的问题

#### 3.1 rank 太小（K=10）

AdaSmooth 使用低秩协方差矩阵 L ∈ R^(d×K)：
- 当前：K = num_queries = 10
- 维度：d = 1000
- 秩比：K/d = 10/1000 = 1%

**问题**：对于 1000 维的 Rosenbrock 函数，秩为 10 的协方差矩阵可能**太低秩**了，无法捕捉足够的方向信息。

#### 3.2 温度衰减太快

```python
beta_t = beta_init / (1 + beta_decay * t)
       = 1.0 / (1 + 0.05 * t)
```

- t=0: beta = 1.0
- t=100: beta = 0.17
- t=1000: beta = 0.02

当 beta 太小时，权重分布变得极度集中在最优样本上，失去了探索能力。

#### 3.3 初始化尺度

```python
# zo.py:859
state['L'] = self.initial_sigma * torch.randn(d, self.K, ...)
#          = 0.05 * randn(1000, 10)
```

初始协方差的尺度是 `0.05 * sqrt(10) ≈ 0.16`，可能对于 Rosenbrock 函数（值域在几十万）来说太小了。

### AdaSmooth 的性能分析

```
AdaSmooth: 381,626  ← 比 ES 稍好，但仍然很差
```

可能的原因：
1. 低秩采样（K=10）在高维（d=1000）下信息不足
2. 温度衰减导致探索-利用失衡
3. 初始尺度不匹配函数的量级

---

## 问题4：为什么 SepCMAES 效果这么好？

### SepCMAES 的优势

```
SepCMAES: 981  ← 最优！
```

SepCMAES 表现最好的原因：

#### 4.1 独立的 Population Size

```python
# utils.py:90
population_size = getattr(args, 'population_size', None)  # None → 自动计算
```

当 population_size = None 时，`cmaes` 库自动设置：
```python
population_size = 4 + int(3 * log(d))
                = 4 + int(3 * log(1000))
                = 4 + int(3 * 6.91)
                = 4 + 20
                = 24
```

**SepCMAES 每次迭代使用 24 个样本**，而其他算法只有 10-11 个！

#### 4.2 对角协方差适应

SepCMAES 学习 d 个独立的步长 σ_i（每维一个），比 xNES 的全协方差更稳定：
- **xNES**：学习 O(d²) 个参数（全协方差矩阵）
- **SepCMAES**：学习 O(d) 个参数（对角元素）
- **AdaSmooth**：学习 O(d*K) 个参数（低秩矩阵）

在 d=1000 时：
- xNES: 需要估计 ~500,000 个参数（不可行）
- SepCMAES: 只需估计 1,000 个参数（可行）
- AdaSmooth: 估计 10,000 个参数（勉强可行）

#### 4.3 不使用外部学习率

SepCMAES 完全不使用配置文件中的 `lr`，而是由 CMA-ES 内部自适应：
```python
# zo.py:741
def step(self, closure):
    # 不使用 lr，直接由 cmaes 库控制更新
    self.cma_optimizer.tell(solutions)
    self._numpy_to_params(self.cma_optimizer._mean)
```

这避免了学习率不匹配的问题。

---

## 修复建议总结

### 立即修复（Critical）

#### 1. 修复 xNES 的学习率

**文件**: `synthetic_and_adversarial/utils.py:84`

```python
# 修改前
return xNES(
    params=params,
    lr=args.lr,  # ← 0.001 太小
    ...
)

# 修改后
return xNES(
    params=params,
    lr=1.0,  # ← 固定为 1.0，xNES 自带步长控制
    ...
)
```

**或者在配置文件中添加**：

```yaml
# config/synthetic.yaml

# 为 xNES 单独设置学习率
xnes_lr: 1.0  # xNES 的自然梯度需要更大的学习率
```

然后在 utils.py 中：
```python
return xNES(
    params=params,
    lr=getattr(args, 'xnes_lr', 1.0),  # 默认 1.0
    ...
)
```

#### 2. 提高 AdaSmooth 的秩

**文件**: `synthetic_and_adversarial/config/synthetic.yaml`

```yaml
# 为 AdaSmooth 单独设置更大的 num_queries
optimizers:
  - adasmooth

adasmooth_num_queries: 64  # 增加低秩矩阵的秩（K）
```

然后在 utils.py 中：
```python
elif name == "adasmooth":
    num_queries_adasmooth = getattr(args, 'adasmooth_num_queries', args.num_queries)
    return AdaSmoothZO(
        params=params,
        lr=1.0,
        num_queries=num_queries_adasmooth,  # 使用独立配置
        ...
    )
```

### 可选优化（Optional）

#### 3. 调整 AdaSmooth 的温度衰减

```yaml
# config/synthetic.yaml
beta_init: 1.0
beta_decay: 0.01  # 减小衰减率（从 0.05 → 0.01）
beta_schedule: polynomial
```

#### 4. ES 使用更小的 mu

由于 ES 没有 baseline，可以尝试更小的扰动：

```yaml
# 为 ES 单独设置
es_mu: 0.01  # 减小扰动（从 0.05 → 0.01）
```

---

## 预期改进效果

应用上述修复后，预期结果：

| 优化器 | 当前损失 | 预期损失 | 改进 |
|--------|---------|---------|------|
| xNES | 443,904 | **~1,000** | ✅ 400x |
| AdaSmooth | 381,626 | **~5,000** | ✅ 70x |
| ES | 457,408 | **~10,000** | ⚠️ 50x（仍不如 Vanilla） |

---

## 验证测试

运行以下命令验证修复：

```bash
cd synthetic_and_adversarial

# 测试修复后的 xNES
FUNC=rosenbrock OPTIMIZER=xnes bash run_script.sh

# 测试修复后的 AdaSmooth
FUNC=rosenbrock OPTIMIZER=adasmooth bash run_script.sh
```

查看优化值是否降到 ~1000 左右（与 SepCMAES 接近）。
