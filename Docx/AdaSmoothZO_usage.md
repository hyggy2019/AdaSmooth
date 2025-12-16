# AdaSmooth-ZO 使用说明

## 📖 简介

AdaSmooth-ZO (Adaptive Smoothing with Low-Rank Updates) 是一个创新的零阶优化算法，通过 KL 正则化策略优化和矩阵匹配来自适应学习搜索均值和低秩协方差。

**核心特性：**
- 自适应低秩采样：x = θ + L·u (u ~ N(0, I_K))
- 自动学习协方差结构（低秩形式）
- 温度调度控制探索-利用权衡
- 复杂度：O(Kd) 时间和空间

## 🎯 核心思想

### 搜索分布

AdaSmooth-ZO 维护高斯策略：π(x) = N(x; θ, LL^T)

- **θ** (theta): 搜索均值（参数中心）
- **L** (d×K): 低秩平滑矩阵
- **LL^T**: 协方差矩阵（秩K）

### KL 正则化更新

每次迭代求解：
```
min_π E_{x~π}[F(x)] + β·KL(π || π_{θ_t, L_t})
```

最优解：
```
π*(x) ∝ π_{θ_t, L_t}(x) · exp(-F(x)/β)
```

### 矩阵匹配

通过加权平均将 π* 投影回高斯族：

**均值更新：**
```
θ_{t+1} = Σ w_k · x_k
其中 w_k = exp(-f_k/β) / Z
```

**协方差更新（低秩）：**
```
L_{t+1} = [√w_1·(x_1 - θ_{t+1}), ..., √w_K·(x_K - θ_{t+1})]
```

这确保了：`L_{t+1}L_{t+1}^T = Σ w_k·(x_k - θ_{t+1})(x_k - θ_{t+1})^T`

---

## 🔧 实现细节

### 两个版本

**1. AdaSmoothZO**
- 适用于单参数模型（synthetic functions）
- 简单直接，易于理解

**2. AdaSmoothZO_MultiParam**
- 适用于多参数模型
- 自动展平/反展平参数
- 统一的协方差学习

### 温度调度

**Polynomial (默认):**
```python
β_t = β_0 / (1 + decay * t)
```

**Exponential:**
```python
β_t = β_0 * exp(-decay * t)
```

**Constant:**
```python
β_t = β_0
```

---

## 🚀 使用方法

### 1. 在配置文件中启用

编辑 `config/synthetic.yaml`：

```yaml
optimizers:
  - vanilla    # 基线
  - adasmooth  # AdaSmooth-ZO（取消注释）
  - zoar
```

### 2. 运行实验

```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml
```

### 3. 自定义参数（可选）

在 `config/synthetic.yaml` 中添加：

```yaml
# AdaSmoothZO 参数
num_queries: 64       # K（批大小/协方差秩）
mu: 0.1               # 初始平滑尺度
beta_init: 1.0        # 初始温度
beta_decay: 0.05      # 温度衰减率
beta_schedule: polynomial  # 调度类型
```

**重要：** AdaSmooth-ZO 强制 `lr=1.0` 和 `update_rule='sgd'`

---

## 📊 参数调优建议

### 批大小 (num_queries = K)

```yaml
num_queries: 32    # 小批量（快速但可能不稳定）
num_queries: 64    # 默认（平衡）
num_queries: 128   # 大批量（稳定但慢）
```

**规则：** K 是协方差的秩，推荐 32-128

### 初始平滑 (mu)

```yaml
mu: 0.05   # 保守（小步长）
mu: 0.1    # 默认
mu: 0.2    # 激进（大步长）
```

**规则：** mu 控制初始探索范围

### 温度参数

**初始温度 (beta_init):**
```yaml
beta_init: 0.5   # 更重视好样本（快速收敛）
beta_init: 1.0   # 默认（平衡）
beta_init: 2.0   # 更均匀探索（鲁棒性）
```

**衰减率 (beta_decay):**
```yaml
beta_decay: 0.01   # 慢衰减（长期探索）
beta_decay: 0.05   # 默认
beta_decay: 0.1    # 快衰减（快速利用）
```

**调度类型 (beta_schedule):**
```yaml
beta_schedule: constant     # 固定温度（持续探索）
beta_schedule: polynomial   # 多项式衰减（默认）
beta_schedule: exponential  # 指数衰减（激进收敛）
```

---

## 💡 使用场景

### ✅ 适合 AdaSmooth-ZO

1. **中高维（d > 100）**
   - 低秩协方差比固定球形更有效
   - K << d 时内存友好

2. **复杂地形**
   - 非凸、多峰函数
   - 温度调度帮助逃离局部最优

3. **需要自适应探索**
   - 早期广泛探索（大L）
   - 后期局部精细化（小L）

4. **参数相关性**
   - 学习低秩协方差结构
   - 捕获关键相关性

### ❌ 不适合的场景

1. **超高维（d > 10000）且 K 大**
   - 内存：O(Kd)
   - K=128, d=10000 需要 ~10MB
   - 考虑减小 K 或用其他方法

2. **极简单函数**
   - 二次函数等
   - Vanilla ES 可能更快

3. **查询成本极高**
   - 每次迭代需要 K 次查询
   - 不复用历史（vs ZoAR）

---

## 🆚 与其他方法对比

| 方法 | 协方差 | 复杂度 | 自适应 | 历史复用 |
|------|--------|--------|--------|---------|
| **Vanilla** | 固定（球形） | O(nd) | ❌ | ❌ |
| **xNES** | 完整（d×d） | O(nd²) | ✅ | ❌ |
| **Sep-CMA-ES** | 对角 | O(nd) | ✅ | ❌ |
| **AdaSmooth-ZO** | **低秩（d×K）** | **O(Kd)** | **✅** | **❌** |
| **ZoAR** | 固定（球形） | O(nd) | ❌ | ✅ |

### AdaSmooth-ZO 的优势

**vs Vanilla:**
- ✅ 自适应协方差（vs 固定）
- ✅ 智能权重（vs 均匀）
- ✅ 温度调度（vs 固定扰动）

**vs xNES:**
- ✅ 低秩：O(Kd) vs O(d²)
- ✅ 可扩展到更高维
- ❌ 只学习秩K结构

**vs ZoAR:**
- ✅ 学习协方差（vs 固定）
- ✅ 自适应权重
- ❌ 不复用历史

---

## 📈 性能预期

### 收敛速度

```
AdaSmooth-ZO ≈ xNES > Sep-CMA-ES > Vanilla
```

**原因：**
- 自适应协方差 + 智能权重
- 温度调度优化探索-利用

### 内存占用 (d=10000, K=64)

```
AdaSmooth-ZO: ~5 MB (d×K)
xNES: ~800 MB (d×d)
Vanilla: ~80 KB (d)
```

### 查询效率

```
每次迭代: K 次查询
不复用历史（vs ZoAR/ReLIZO）
```

---

## 📝 示例

### Rosenbrock 函数（d=100）

```yaml
# config/synthetic.yaml
func_name: rosenbrock
dimension: 100
num_iterations: 5000

optimizers:
  - vanilla    # 基线
  - adasmooth  # AdaSmooth-ZO
  - xnes       # xNES

# AdaSmooth-ZO 参数
num_queries: 64
mu: 0.1
beta_init: 1.0
beta_decay: 0.05
beta_schedule: polynomial
```

**预期结果：**
- **AdaSmooth-ZO**: 快速收敛，学习低秩结构
- **xNES**: 收敛最快（完整协方差）
- **Vanilla**: 收敛最慢（固定协方差）

### Levy 函数（d=1000）

```yaml
func_name: levy
dimension: 1000
num_iterations: 10000

optimizers:
  - vanilla
  - adasmooth
  - zoar

# AdaSmooth-ZO 参数
num_queries: 128  # 更大 K
mu: 0.05
beta_init: 2.0    # 更大温度（更多探索）
beta_decay: 0.03  # 慢衰减
```

**预期结果：**
- **AdaSmooth-ZO**: 自适应探索，好收敛
- **ZoAR**: 查询效率高（历史复用）
- **Vanilla**: 固定协方差，较慢

---

## ⚠️ 注意事项

### 1. 强制 lr=1.0 和 update_rule='sgd'

AdaSmooth-ZO 使用伪梯度技巧：

```python
# 设置：grad = (θ_old - θ_new) / lr
# SGD 步骤：θ ← θ - lr·grad = θ_new ✓
```

因此 **必须** `lr=1.0` 和 `update_rule='sgd'`

### 2. 温度调度的重要性

温度 β 控制探索-利用权衡：

- **β 大**: 权重接近均匀 → 更多探索
- **β 小**: 权重集中于好样本 → 更多利用

推荐使用衰减调度（polynomial 或 exponential）

### 3. K 的选择

K 是协方差的秩：

- **K 小**: 内存少，但可能无法捕获复杂结构
- **K 大**: 捕获更多信息，但内存和时间增加

推荐：32 ≤ K ≤ 128

### 4. 与其他优化器的接口

AdaSmooth-ZO 不使用传统梯度：

```python
# 内部设置 param.grad 是为了兼容 SGD 接口
# 实际更新通过加权平均完成
```

---

## 🔍 技术细节

### 低秩采样

```python
# 采样 u ~ N(0, I_K) 在 K 维潜在空间
u = torch.randn(K)

# 映射到 d 维参数空间
x = theta + L @ u  # O(Kd) 复杂度
```

### 权重计算

```python
# 指数权重
v = torch.exp(-f_values / beta)

# 归一化
w = v / v.sum()
```

### 均值更新

```python
# 加权平均
theta_new = torch.sum(w.unsqueeze(1) * X, dim=0)
```

### 协方差更新（低秩分解）

```python
# 残差
residuals = X - theta_new  # (K, d)

# 加权并转置
L_new = (torch.sqrt(w).unsqueeze(1) * residuals).T  # (d, K)

# 验证：L_new @ L_new.T ≈ Σ w_k·(x_k - theta_new)(x_k - theta_new).T
```

---

## 🧪 调试建议

### 检查收敛曲线

```python
import torch
import matplotlib.pyplot as plt

# 加载结果
history = torch.load('results/synthetic/levy_adasmooth_...')

# 绘制
plt.plot(history)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('AdaSmooth-ZO Convergence')
plt.show()
```

### 检查权重分布

```python
# AdaSmooth-ZO 跟踪权重历史
weights_history = optimizer.history['weights']

# 绘制权重熵
import numpy as np
entropy = [-np.sum(w * np.log(w + 1e-10)) for w in weights_history]
plt.plot(entropy)
plt.title('Weight Entropy over Time')
plt.show()
```

### 检查 L 范数

```python
# AdaSmooth-ZO 跟踪 L 范数
L_norms = optimizer.history['L_norms']

plt.plot(L_norms)
plt.title('||L|| over Time')
plt.ylabel('Frobenius Norm')
plt.show()
```

**正常行为：**
- 初期：L 范数较大（广泛探索）
- 后期：L 范数减小（局部精细化）

### 检查温度

```python
# 温度历史
beta_history = optimizer.history['beta']

plt.plot(beta_history)
plt.title('Temperature β over Time')
plt.show()
```

---

## 📚 参考资料

1. **KL-Regularized Policy Optimization:**
   - Natural Evolution Strategies
   - Relative Entropy Policy Search

2. **Low-Rank Optimization:**
   - CMA-ES with Selective Covariance
   - Sketched Newton Methods

3. **Temperature Annealing:**
   - Simulated Annealing
   - Boltzmann Exploration

---

## ✅ 总结

### 何时使用 AdaSmooth-ZO

- ✅ 中高维（d > 100）
- ✅ 复杂非凸函数
- ✅ 需要自适应协方差
- ✅ 内存受限（vs xNES）

### 何时不用 AdaSmooth-ZO

- ❌ 超简单函数（二次函数）
- ❌ 查询极其昂贵（考虑 ZoAR）
- ❌ d < 50（Vanilla 可能更快）

### 推荐配置

```yaml
optimizers:
  - vanilla      # 简单基线
  - adasmooth    # 自适应低秩
  - zoar         # 查询复用

# AdaSmooth-ZO 参数
num_queries: 64       # K
mu: 0.1               # 初始平滑
beta_init: 1.0        # 初始温度
beta_decay: 0.05      # 衰减率
beta_schedule: polynomial
```

**AdaSmooth-ZO 在需要自适应探索的中高维优化问题上表现优异！** 🚀

它通过低秩协方差学习和智能温度调度，实现了探索-利用的最佳平衡，同时保持 O(Kd) 的高效复杂度。
