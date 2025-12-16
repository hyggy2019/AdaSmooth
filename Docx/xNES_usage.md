# xNES (Exponential Natural Evolution Strategies) 使用说明

## 📖 简介

xNES 是自然进化策略（Natural Evolution Strategies, NES）的一个变体，通过自适应协方差矩阵来高效探索参数空间。

**论文：** [Natural Evolution Strategies](https://arxiv.org/abs/1106.4487)

## 🎯 核心思想

xNES 维护一个搜索分布 `N(μ, σ²B^T B)`，其中：
- **μ** (mu): 分布的均值（搜索中心）
- **σ** (sigma): 全局步长（标量）
- **B** (B matrix): 协方差矩阵的形状部分（矩阵）

通过**自然梯度**更新这三个参数，使搜索分布逐步逼近最优解。

---

## 🔬 数学原理

### 搜索分布

```
z ~ N(μ, C)
其中 C = σ² B^T B
```

### 参数更新

使用自然梯度更新：

**1. 均值 μ：**
```
μ ← μ + η_μ σ B ∇_δ
```

**2. 步长 σ：**
```
σ ← σ · exp(0.5 η_σ ∇_σ)
```

**3. 形状矩阵 B：**
```
B ← B · exp(0.5 η_B ∇_B)
```

其中梯度计算为：
```
∇_δ = Σ u_i s_i
∇_σ = trace(M) / d
∇_B = M - ∇_σ I
M = Σ u_i s_i s_i^T - (Σ u_i) I
```

`s_i` 是标准正态样本，`u_i` 是权重（可以是 fitness shaping 后的）。

---

## 🆚 与其他ES方法对比

| 方法 | 协方差矩阵 | 自然梯度 | 自适应 | 复杂度 |
|------|-----------|---------|--------|--------|
| **ES (纯)** | 固定（球形） | ❌ | ❌ | O(d) |
| **Vanilla** | 固定（球形） | ❌ | ❌ | O(d) |
| **RL** | 固定（球形） | ❌ | ❌ | O(d) |
| **xNES** | 自适应（全） | ✅ | ✅ | O(d²) |
| **CMA-ES** | 自适应（全） | ❌ | ✅ | O(d²) |

**关键优势：**
- ✅ 自适应协方差矩阵（处理相关性）
- ✅ 自然梯度（更快收敛）
- ✅ Fitness shaping（降低方差）
- ❌ 计算复杂度 O(d²)（高维时较慢）

---

## 🔧 实现细节

### 初始化

```python
class xNES(ZerothOrderOptimizer):
    def __init__(self, params, lr=0.001, num_queries=10, mu=0.01,
                 eta_mu=1.0, eta_sigma=None, eta_bmat=None,
                 use_fshape=True):
        # mu: 初始步长（后用作 sigma）
        # eta_mu: 均值学习率
        # eta_sigma: sigma 学习率（自动计算）
        # eta_bmat: B 矩阵学习率（自动计算）
        # use_fshape: 是否使用 fitness shaping
```

### 默认学习率

如果未指定，使用以下公式（来自论文）：

```python
eta_sigma = 3(3 + log(d)) / (5d√d)
eta_bmat = 3(3 + log(d)) / (5d√d)
```

### Fitness Shaping

为降低方差，使用基于排序的权重：

```python
a = log(1 + 0.5n)
u_i = max(0, a - log(k)) / Σ
u_i ← u_i - 1/n  # 中心化
```

按适应度排序后，最好的样本权重最高。

---

## 🚀 使用方法

### 1. 在配置文件中启用

编辑 `config/synthetic.yaml`：

```yaml
optimizers:
  - vanilla  # 对比基线
  - xnes     # 启用 xNES（取消注释）
  - zoar
```

### 2. 运行实验

```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml
```

### 3. 调整 xNES 参数（可选）

在 `config/synthetic.yaml` 中添加：

```yaml
# xNES 参数（可选）
eta_mu: 1.0         # 均值学习率
eta_sigma: null     # sigma 学习率（null 表示自动）
eta_bmat: null      # B 矩阵学习率（null 表示自动）
use_fshape: true    # 使用 fitness shaping
initial_sigma: 0.1  # 初始步长（默认 0.1）
```

---

## 📊 参数调优建议

### 学习率 (lr)

**xNES 默认 lr = 1.0**（与其他方法不同）

xNES 使用自然梯度和自适应步长，因此：
```yaml
lr: 1.0  # 推荐（xNES 默认）
```

如需调整：
```yaml
lr: 0.5  # 保守
lr: 2.0  # 激进
```

**注意：** xNES **必须使用 SGD**，不支持 Adam/RadAZO（内部已有自适应机制）

### 初始步长 (initial_sigma)

```yaml
initial_sigma: 0.05   # 保守（小步长）
initial_sigma: 0.1    # 默认（推荐）
initial_sigma: 0.5    # 激进（大步长）
```

**规则：** 设为解空间范围的 10-30%

### 种群大小 (num_queries)

```yaml
num_queries: 10   # 默认（低维）
num_queries: 20   # 中维（d ~ 100-1000）
num_queries: 50   # 高维（d > 1000）
```

**规则：** 论文建议 `4 + 3·log(d)`

### 学习率

**保守设置：**
```yaml
eta_mu: 0.5
eta_sigma: 0.01
eta_bmat: 0.01
```

**激进设置：**
```yaml
eta_mu: 1.0
eta_sigma: null  # 使用默认（自动计算）
eta_bmat: null
```

### Fitness Shaping

```yaml
use_fshape: true   # 推荐：降低方差
use_fshape: false  # 不推荐：除非需要原始适应度
```

---

## 💡 使用场景

### ✅ 适合 xNES 的场景

1. **中等维度（d ~ 10-1000）**
   - 协方差矩阵 O(d²) 可接受

2. **变量相关性强**
   - xNES 能学习协方差结构
   - 例：Rosenbrock 函数

3. **需要快速收敛**
   - 自然梯度加速收敛

4. **函数平滑**
   - xNES 假设局部平滑

### ❌ 不适合的场景

1. **超高维（d > 10000）**
   - 内存和计算成本 O(d²)
   - 建议用 Vanilla 或 ZoAR

2. **离散/非平滑函数**
   - 协方差矩阵无法有效学习

3. **查询成本极高**
   - xNES 不复用历史查询
   - 建议用 ZoAR

---

## 📈 性能预期

### 收敛速度（相同查询预算）

```
xNES > Vanilla > ES (纯)
```

**原因：**
- 自适应协方差矩阵
- 自然梯度更新

### 方差（估计稳定性）

```
xNES < RL < Vanilla < ES (纯)
```

**原因：**
- Fitness shaping 降低方差
- 协方差矩阵优化采样方向

### 查询成本（每次迭代）

```
xNES = Vanilla = n + 1
```

但 xNES 计算复杂度更高（O(d²）矩阵运算）。

---

## 🔍 与 CMA-ES 对比

| 特性 | xNES | CMA-ES |
|------|------|--------|
| **梯度类型** | 自然梯度 | 进化路径 |
| **协方差更新** | 指数映射 | 加性更新 |
| **理论基础** | 信息几何 | 启发式 |
| **收敛速度** | 快 | 中等 |
| **鲁棒性** | 中等 | 高 |
| **实现复杂度** | 中等 | 高 |

**xNES 优势：**
- 理论上更优雅（基于自然梯度）
- 更简洁的更新规则

**CMA-ES 优势：**
- 更成熟（广泛应用）
- 更鲁棒（内置各种技巧）

---

## 📝 示例对比

### Levy 函数（d=10000）

```yaml
# config/synthetic.yaml
func_name: levy
dimension: 10000
num_iterations: 20000

optimizers:
  - vanilla  # 基线
  - xnes     # xNES
  - zoar     # ZoAR
```

**预期结果：**
- **xNES**: 收敛最快，最终精度高
- **Vanilla**: 收敛中等
- **ZoAR**: 查询效率最高（复用历史）

### Rastrigin 函数（高度多峰）

```yaml
func_name: rastrigin
dimension: 10000
num_iterations: 20000

optimizers:
  - rl       # 排序变换（抗异常值）
  - xnes     # 自适应协方差
  - zoar     # 历史平滑
```

**预期结果：**
- **RL**: 最鲁棒（排序降低局部极值影响）
- **xNES**: 收敛快（如果不陷入局部最优）
- **ZoAR**: 查询效率最高

---

## ⚠️ 注意事项

### 1. 内存占用

xNES 需要存储 `d×d` 的 B 矩阵：

```python
memory = d² × 8 bytes (float64)
```

**示例：**
- d=1000: ~8 MB
- d=10000: ~800 MB
- d=100000: ~80 GB（不可行）

### 2. 计算复杂度

每次迭代：
- 矩阵乘法：`O(d²)`
- 矩阵指数：`O(d³)`（但通常优化到 O(d²)）

**建议：** d < 5000 较为实用

### 3. 与其他优化器冲突

xNES 内部管理参数更新，**不使用** `update_rule` 参数：

```python
# xNES 内部使用自己的更新规则
# update_rule='sgd' 被忽略
```

---

## 🧪 调试技巧

### 检查 sigma 变化

```python
# xNES 内部维护 sigma_xnes
# 可以在代码中打印查看
print(f"Current sigma: {optimizer.sigma_xnes}")
```

**正常行为：**
- 初期：sigma 较大（广泛探索）
- 后期：sigma 逐渐减小（局部精细搜索）

### 检查 B 矩阵

```python
# B 应该保持正交性（近似）
B = optimizer.bmat
print(f"B^T B 对角线: {torch.diag(B.T @ B)}")
# 应该接近全1（正交）
```

### Fitness 曲线

xNES 应该表现出稳定下降：

```python
import matplotlib.pyplot as plt
history = torch.load('results/synthetic/levy_xnes_...')
plt.plot(history)
plt.yscale('log')
plt.title('xNES Optimization')
plt.show()
```

---

## 📚 参考资料

1. **原始论文：**
   - Glasmachers et al. (2010) "Exponential Natural Evolution Strategies"
   - [https://arxiv.org/abs/1106.4487](https://arxiv.org/abs/1106.4487)

2. **自然梯度：**
   - Amari (1998) "Natural Gradient Works Efficiently in Learning"

3. **进化策略综述：**
   - Hansen (2016) "The CMA Evolution Strategy: A Tutorial"

---

## ✅ 总结

### 何时使用 xNES

- ✅ 中等维度（10 < d < 5000）
- ✅ 变量相关性强
- ✅ 需要快速收敛
- ✅ 函数较平滑

### 何时不用 xNES

- ❌ 超高维（d > 10000）
- ❌ 查询极其昂贵
- ❌ 函数极度多峰/非平滑

### 推荐配置

```yaml
optimizers:
  - vanilla   # 简单基线
  - xnes      # 自适应协方差
  - zoar      # 查询复用

num_queries: 20  # xNES 推荐稍大种群
use_fshape: true # 使用 fitness shaping
```

**xNES 是介于简单 ES 和复杂 CMA-ES 之间的优秀选择！** 🚀
