# Sep-CMA-ES (Separable Covariance Matrix Adaptation Evolution Strategy) 使用说明

## 📖 简介

Sep-CMA-ES 是 CMA-ES 的一个变体，将协方差矩阵限制为对角形式，大幅提升了在高维优化任务中的可扩展性。

**论文：** Ros and Hansen (2008) - "A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity"

## 🎯 核心思想

与完整的 CMA-ES 不同，Sep-CMA-ES：
- 使用**对角协方差矩阵**而非完整协方差矩阵
- 将参数数量从 O(d²) 降低到 O(d)
- 提高协方差矩阵的学习率
- 在**可分离函数**上性能优于完整 CMA-ES

## 🔬 数学原理

### 搜索分布

```
z ~ N(μ, C)
其中 C = diag(σ₁², σ₂², ..., σ_d²)
```

与完整 CMA-ES 的 C = σ² B^T B 相比，Sep-CMA-ES 使用对角矩阵。

### 参数更新

Sep-CMA-ES 维护：
- **μ** (mean): 搜索分布的均值（d维向量）
- **σ** (sigma): 每个维度的标准差（d维向量）

通过进化路径和排序样本更新这些参数。

---

## 🆚 与其他方法对比

| 方法 | 协方差矩阵 | 参数数量 | 内存 | 适用维度 |
|------|-----------|---------|------|----------|
| **ES (Vanilla)** | 固定（球形） | O(d) | O(d) | 任意 |
| **xNES** | 自适应（完整） | O(d²) | O(d²) | d < 5000 |
| **Sep-CMA-ES** | 自适应（对角） | O(d) | O(d) | **d > 5000** |
| **CMA-ES** | 自适应（完整） | O(d²) | O(d²) | d < 5000 |

**关键优势：**
- ✅ 对角协方差矩阵（线性复杂度 O(d)）
- ✅ 适合超高维优化（d > 10000）
- ✅ 在可分离函数上性能优于完整 CMA-ES
- ✅ 使用成熟的 `cmaes` 库实现
- ❌ 无法学习变量间的相关性

---

## 🔧 实现细节

### 使用 cmaes 库

本实现基于 `cmaes` 库的 `SepCMA` 类：

```python
from cmaes import SepCMA

class SepCMAES(torch.optim.Optimizer):
    """Wraps cmaes.SepCMA for PyTorch optimizer interface."""

    def __init__(self, params, lr=0.001, sigma=0.1, population_size=None):
        # Initialize SepCMA optimizer
        self.cma_optimizer = SepCMA(mean=initial_params, sigma=sigma)

    def step(self, closure):
        # Ask-Tell interface
        solutions = []
        for _ in range(self.cma_optimizer.population_size):
            x = self.cma_optimizer.ask()
            loss = evaluate(x)
            solutions.append((x, loss))

        self.cma_optimizer.tell(solutions)
```

### 默认参数

- **sigma**: 初始步长（默认使用 `mu` 参数值）
- **population_size**: 种群大小（默认：4 + 3·log(d)）

---

## 🚀 使用方法

### 1. 在配置文件中启用

编辑 `config/synthetic.yaml`：

```yaml
optimizers:
  - vanilla    # 对比基线
  - sepcmaes   # 启用 Sep-CMA-ES（取消注释）
  - zoar
```

### 2. 运行实验

```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml
```

### 3. 调整参数（可选）

在 `config/synthetic.yaml` 中添加：

```yaml
# Sep-CMA-ES 参数（可选）
sigma: 0.1           # 初始步长
population_size: 20  # 种群大小（null 表示自动）
```

---

## 📊 参数调优建议

### 初始步长 (sigma)

```yaml
sigma: 0.05   # 保守（小步长）
sigma: 0.1    # 默认
sigma: 0.5    # 激进（大步长）
```

**规则：**
- 如果知道解的大致范围，设置为该范围的 1/3
- 否则使用 `mu` 参数值作为默认

### 种群大小 (population_size)

```yaml
population_size: null  # 自动：4 + 3·log(d)
population_size: 10    # 小种群（快速但可能不稳定）
population_size: 50    # 大种群（慢但更稳定）
```

**规则：**
- 低维（d < 100）：10-20
- 中维（d ~ 1000）：20-50
- 高维（d > 10000）：使用默认（自动计算）

---

## 💡 使用场景

### ✅ 适合 Sep-CMA-ES 的场景

1. **超高维（d > 5000）**
   - 完整 CMA-ES 和 xNES 内存/计算过大
   - Sep-CMA-ES 保持 O(d) 复杂度

2. **可分离函数**
   - 变量之间独立或弱相关
   - 例：Ackley、Rastrigin（在某些变换下）
   - Sep-CMA-ES 可能优于完整 CMA-ES

3. **内存受限**
   - d² 矩阵无法放入内存
   - 例：d=100000 需要 ~80GB（完整 CMA-ES）vs ~1MB（Sep-CMA-ES）

4. **需要自适应但不需要学习相关性**
   - 每个维度独立优化步长
   - 比固定球形协方差（Vanilla ES）更灵活

### ❌ 不适合的场景

1. **强变量相关性**
   - 例：Rosenbrock 函数（相邻维度高度相关）
   - 建议用 xNES 或完整 CMA-ES

2. **低维（d < 100）**
   - 完整 CMA-ES 或 xNES 更合适
   - Sep-CMA-ES 的优势不明显

3. **查询成本极高**
   - CMA-ES 每次迭代需要评估整个种群
   - 不复用历史查询
   - 建议用 ZoAR

---

## 📈 性能预期

### 复杂度对比

| 方法 | 每次迭代时间 | 内存 | 可扩展性 |
|------|------------|------|----------|
| Vanilla | O(nd) | O(d) | ⭐⭐⭐⭐⭐ |
| xNES | O(nd²) | O(d²) | ⭐⭐⭐ |
| Sep-CMA-ES | O(nd) | O(d) | ⭐⭐⭐⭐⭐ |
| ZoAR | O(nd + kd) | O(kd) | ⭐⭐⭐⭐ |

**说明：**
- n: 种群大小（通常 4 + 3·log(d)）
- d: 参数维度
- k: 历史数量

### 收敛速度（可分离函数）

```
Sep-CMA-ES > xNES ≈ CMA-ES > Vanilla > ES
```

**原因：**
- Sep-CMA-ES 自适应每个维度的步长
- 在可分离函数上与完整 CMA-ES 性能相当

### 收敛速度（不可分离函数）

```
xNES ≈ CMA-ES > Sep-CMA-ES > Vanilla
```

**原因：**
- Sep-CMA-ES 无法学习变量相关性
- 完整协方差矩阵更有优势

---

## 🔍 与 xNES 对比

| 特性 | Sep-CMA-ES | xNES |
|------|-----------|------|
| **协方差矩阵** | 对角（分离） | 完整 |
| **参数数量** | O(d) | O(d²) |
| **内存占用** | O(d) | O(d²) |
| **最大维度** | 无限制 | ~5000 |
| **可分离函数** | 优秀 | 优秀 |
| **不可分离函数** | 中等 | 优秀 |
| **实现来源** | cmaes 库 | 自定义 |

**何时选择 Sep-CMA-ES：**
- ✅ d > 5000（xNES 内存不足）
- ✅ 函数可分离
- ✅ 需要稳定的库实现

**何时选择 xNES：**
- ✅ d < 5000
- ✅ 强变量相关性
- ✅ 需要最快收敛

---

## 📝 示例对比

### 高维可分离函数（Levy, d=50000）

```yaml
# config/synthetic.yaml
func_name: levy
dimension: 50000
num_iterations: 5000

optimizers:
  # - xnes       # 内存不足（需要 ~20GB）
  - sepcmaes     # Sep-CMA-ES（~400KB）✅
  - vanilla      # 基线
  - zoar         # 查询复用
```

**预期结果：**
- **Sep-CMA-ES**: 收敛最快，内存可接受
- **Vanilla**: 收敛较慢
- **ZoAR**: 查询效率最高
- **xNES**: 无法运行（内存溢出）

### 低维不可分离函数（Rosenbrock, d=100）

```yaml
func_name: rosenbrock
dimension: 100
num_iterations: 10000

optimizers:
  - xnes       # xNES（学习相关性）✅
  - sepcmaes   # Sep-CMA-ES
  - vanilla    # 基线
```

**预期结果：**
- **xNES**: 收敛最快（学习旋转）
- **Sep-CMA-ES**: 收敛中等
- **Vanilla**: 收敛最慢

---

## ⚠️ 注意事项

### 1. 不使用梯度

Sep-CMA-ES 完全基于函数评估，不计算梯度：

```python
# CMA-ES 内部处理参数更新
# update_rule 参数被忽略
```

### 2. 种群评估

每次迭代需要评估整个种群：

```python
queries_per_iteration = population_size  # 通常 4 + 3·log(d)
```

对于 d=10000，种群约为 34，每次迭代 34 次查询。

### 3. 与其他优化器的接口差异

Sep-CMA-ES 继承自 `torch.optim.Optimizer` 但不是 `ZerothOrderOptimizer`：
- 不使用 `num_queries` 参数（使用 `population_size`）
- 不使用 `update_rule` 参数（内部自适应）
- 使用 `sigma` 而非 `mu` 作为步长

---

## 🧪 调试技巧

### 检查收敛

```python
import torch
import matplotlib.pyplot as plt

history = torch.load('results/synthetic/levy_sepcmaes_...')
plt.plot(history)
plt.yscale('log')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Sep-CMA-ES Optimization')
plt.show()
```

**正常行为：**
- 初期：快速下降
- 中期：平稳优化
- 后期：精细收敛

### 检查步长

可以通过 `cmaes` 库的接口查看：

```python
# 在优化器中添加打印
print(f"Generation {gen}: sigma = {self.cma_optimizer._sigma}")
```

**正常行为：**
- sigma 逐渐自适应调整
- 不会过快衰减到 0

### 检查种群大小

```python
print(f"Population size: {optimizer.cma_optimizer.population_size}")
```

对于 d=10000，应该约为 30-40。

---

## 📚 参考资料

1. **原始论文：**
   - Ros & Hansen (2008) "A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity"

2. **CMA-ES 综述：**
   - Hansen (2016) "The CMA Evolution Strategy: A Tutorial"

3. **cmaes 库文档：**
   - https://github.com/CyberAgentAILab/cmaes

---

## ✅ 总结

### 何时使用 Sep-CMA-ES

- ✅ 超高维（d > 5000）
- ✅ 可分离函数
- ✅ 内存受限
- ✅ 需要稳定库实现

### 何时不用 Sep-CMA-ES

- ❌ 低维（d < 100）
- ❌ 强变量相关性
- ❌ 查询极其昂贵（用 ZoAR）

### 推荐配置

```yaml
optimizers:
  - sepcmaes    # 高维优化
  - vanilla     # 简单基线
  - zoar        # 查询复用

# Sep-CMA-ES 参数
sigma: 0.1           # 初始步长
population_size: null # 自动计算
```

**Sep-CMA-ES 是超高维优化的最佳选择！** 🚀

在 d > 10000 时，Sep-CMA-ES 提供了自适应优化的优势，同时保持线性的内存和计算复杂度。
