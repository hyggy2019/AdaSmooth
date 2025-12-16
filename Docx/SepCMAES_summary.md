# Sep-CMA-ES 实现总结

## ✅ 完成的工作

### 1. 实现 Sep-CMA-ES 优化器

**文件：** `synthetic_and_adversarial/optimizer/zo.py`

**类：** `SepCMAES(torch.optim.Optimizer)` (第 661-778 行)

**核心功能：**
- ✅ 基于 `cmaes` 库的 `SepCMA` 实现
- ✅ 对角协方差矩阵（O(d) 复杂度）
- ✅ PyTorch-NumPy 转换接口
- ✅ Ask-Tell 优化模式
- ✅ 自动种群大小计算

---

## 🎯 核心算法

### 参数

Sep-CMA-ES 使用 `cmaes.SepCMA` 库，维护：

1. **μ (mean)**: 搜索分布的均值（d维向量）
2. **σ (sigma)**: 每个维度的标准差（d维向量，对角协方差）
3. **进化路径**: 累积的搜索方向信息

### 优化流程

```python
# 1. 初始化
optimizer = SepCMA(mean=initial_params, sigma=sigma)

# 2. 每次迭代
for generation in range(max_generations):
    # Ask: 生成候选解
    solutions = []
    for _ in range(population_size):
        x = optimizer.ask()
        f_x = evaluate(x)
        solutions.append((x, f_x))

    # Tell: 更新分布
    optimizer.tell(solutions)

    # 检查终止条件
    if optimizer.should_stop():
        break
```

---

## 📁 文件修改

### 新增/修改的文件

1. **`synthetic_and_adversarial/optimizer/zo.py`**
   - 添加导入：`import numpy as np`, `from cmaes import SepCMA`
   - 添加 `SepCMAES` 类（115行）

2. **`synthetic_and_adversarial/utils.py`**
   - 导入 `SepCMAES`
   - 在 `get_optimizer()` 中注册 "sepcmaes"
   - 支持可选参数：sigma, population_size

3. **`synthetic_and_adversarial/config/synthetic.yaml`**
   - 在优化器列表中添加 sepcmaes（注释）
   - 添加 Sep-CMA-ES 参数说明

4. **`synthetic_and_adversarial/config/adversarial.yaml`**
   - 在优化器列表中添加 sepcmaes（注释）

5. **`synthetic_and_adversarial/config/synthetic-baseline.yaml`**
   - 在优化器列表中添加 sepcmaes（注释）

6. **`Docx/SepCMAES_usage.md`** ✨ 新建
   - 完整的使用说明
   - 数学原理
   - 参数调优建议
   - 使用场景分析
   - 与 xNES 对比

7. **`Docx/SepCMAES_summary.md`** ✨ 新建
   - 实现总结（本文档）

---

## 🔧 使用方法

### 1. 在配置文件中启用

编辑 `config/synthetic.yaml`：

```yaml
optimizers:
  - vanilla    # 基线
  - sepcmaes   # 取消注释启用 Sep-CMA-ES
  - zoar
```

### 2. 运行实验

```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml
```

### 3. 高级配置（可选）

在 `config/synthetic.yaml` 中添加：

```yaml
# Sep-CMA-ES 特定参数
sigma: 0.1           # 初始步长
population_size: 20  # 种群大小（null = 自动）
```

---

## 📊 Sep-CMA-ES vs 其他方法

### 计算复杂度

| 方法 | 每次迭代 | 内存 | 最大维度 |
|------|---------|------|---------|
| Vanilla | O(nd) | O(d) | 无限制 |
| xNES | O(nd²) | O(d²) | ~5000 |
| **Sep-CMA-ES** | **O(nd)** | **O(d)** | **无限制** |
| ZoAR | O(nd + kd) | O(kd) | 无限制 |

**说明：**
- n: 种群大小（通常 4 + 3·log(d)）
- d: 参数维度
- k: 历史数量

### 收敛性能

**可分离函数：**
```
Sep-CMA-ES ≈ xNES > ZoAR > Vanilla
```

**不可分离函数：**
```
xNES > Sep-CMA-ES > ZoAR > Vanilla
```

**原因：**
- Sep-CMA-ES: 自适应对角协方差（每维独立优化）
- xNES: 自适应完整协方差（学习变量相关性）
- ZoAR: 查询复用
- Vanilla: 固定球形协方差

### 内存占用

**d=10000 时：**

| 方法 | 内存 |
|------|------|
| Vanilla | ~80 KB |
| ZoAR (k=5) | ~400 KB |
| **Sep-CMA-ES** | **~80 KB** |
| xNES | **~800 MB** ❌ |

---

## 💡 适用场景

### ✅ 适合 Sep-CMA-ES

1. **超高维（d > 5000）**
   - xNES 和完整 CMA-ES 内存/计算过大
   - Sep-CMA-ES 保持 O(d) 复杂度

2. **可分离函数**
   - Levy, Ackley 等（变量独立）
   - Sep-CMA-ES 可能优于完整 CMA-ES

3. **内存受限环境**
   - d² 矩阵无法放入内存

4. **需要成熟库支持**
   - `cmaes` 库经过充分测试

### ❌ 不适合 Sep-CMA-ES

1. **低维（d < 100）**
   - xNES 或完整 CMA-ES 更合适
   - Sep-CMA-ES 优势不明显

2. **强变量相关性**
   - 例：Rosenbrock（相邻维度高度相关）
   - 建议用 xNES

3. **查询成本极高**
   - CMA-ES 每次迭代需评估整个种群
   - 不复用历史查询
   - 建议用 ZoAR

---

## 🆚 与 xNES 详细对比

| 特性 | Sep-CMA-ES | xNES |
|------|-----------|------|
| **协方差类型** | 对角（分离） | 完整（d×d 矩阵） |
| **参数数量** | O(d) | O(d²) |
| **内存占用** | O(d) | O(d²) |
| **计算复杂度** | O(nd) | O(nd²) |
| **理论基础** | 进化路径 | 自然梯度 |
| **学习相关性** | ❌ | ✅ |
| **可分离函数** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **不可分离函数** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **最大维度** | 无限制 | ~5000 |
| **实现来源** | cmaes 库 | 自定义 |
| **成熟度** | 工业级 | 研究级 |

**选择建议：**

**使用 Sep-CMA-ES 当：**
- d > 5000
- 函数可分离
- 内存受限
- 需要稳定的工业级实现

**使用 xNES 当：**
- d < 5000
- 强变量相关性
- 追求最快收敛
- 理论研究

---

## 🔍 技术细节

### PyTorch-NumPy 转换

```python
def _params_to_numpy(self) -> np.ndarray:
    """Convert PyTorch parameters to numpy vector."""
    all_params = []
    for group in self.param_groups:
        for param in group['params']:
            all_params.append(param.detach().cpu().numpy().flatten())
    return np.concatenate(all_params)

def _numpy_to_params(self, x: np.ndarray):
    """Convert numpy vector back to PyTorch parameters."""
    offset = 0
    for param, shape, numel in zip(...):
        param_data = x[offset:offset+numel].reshape(shape)
        param.data = torch.from_numpy(param_data).to(param.device, param.dtype)
        offset += numel
```

### Ask-Tell 接口

```python
@torch.no_grad()
def step(self, closure):
    solutions = []

    # Ask phase: generate candidates
    for _ in range(self.cma_optimizer.population_size):
        x = self.cma_optimizer.ask()
        self._numpy_to_params(x)
        loss = closure()
        solutions.append((x, loss.item()))

    # Tell phase: update distribution
    self.cma_optimizer.tell(solutions)

    # Update parameters to current mean
    self._numpy_to_params(self.cma_optimizer._mean)

    return min(loss for _, loss in solutions)
```

### 默认种群大小

`cmaes` 库使用：
```python
population_size = 4 + floor(3 * log(d))
```

**示例：**
- d=10: population = 10
- d=100: population = 17
- d=1000: population = 24
- d=10000: population = 31

---

## 📈 性能预期

### Levy 函数（d=50000）

```yaml
# config/synthetic.yaml
func_name: levy
dimension: 50000
num_iterations: 5000

optimizers:
  - sepcmaes   # Sep-CMA-ES
  - vanilla    # 基线
  - zoar       # ZoAR
```

**预期结果：**
- **Sep-CMA-ES**: 收敛最快（自适应步长）
- **ZoAR**: 查询效率最高（复用历史）
- **Vanilla**: 收敛最慢（固定步长）

**内存使用：**
- Sep-CMA-ES: ~400 KB
- ZoAR: ~2 MB
- Vanilla: ~400 KB

### Rastrigin 函数（高度多峰）

```yaml
func_name: rastrigin
dimension: 10000
num_iterations: 20000

optimizers:
  - sepcmaes   # 自适应步长
  - rl         # 排序变换
  - zoar       # 历史平滑
```

**预期结果：**
- **RL**: 最鲁棒（排序降低局部极值影响）
- **Sep-CMA-ES**: 收敛快（如果避开局部最优）
- **ZoAR**: 稳定性中等

---

## ⚠️ 已知限制

### 1. 无法学习变量相关性

Sep-CMA-ES 使用对角协方差矩阵：
- 只能优化每个维度的步长
- 无法学习变量间的旋转/缩放关系

**影响：**
- Rosenbrock 等不可分离函数性能下降

### 2. 每次迭代查询数较多

```python
queries_per_iteration = population_size ≈ 4 + 3·log(d)
```

**示例：**
- d=10000: 每次迭代约 31 次查询
- 相比之下，Vanilla/ZoAR 每次约 10 次查询

### 3. 不复用历史查询

与 ZoAR/ReLIZO 不同，CMA-ES 不复用之前的查询结果。

---

## 🧪 调试建议

### 检查收敛曲线

```python
import torch
import matplotlib.pyplot as plt

history = torch.load('results/synthetic/levy_sepcmaes_...')
plt.plot(history)
plt.yscale('log')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Sep-CMA-ES Convergence')
plt.show()
```

**正常行为：**
- 指数衰减（前期快速下降）
- 稳定收敛（后期缓慢优化）

### 检查种群大小

```python
print(f"Population size: {optimizer.cma_optimizer.population_size}")
```

应该约为 `4 + 3·log(d)`。

### 对比基线方法

```yaml
optimizers:
  - vanilla    # 固定步长基线
  - sepcmaes   # 自适应步长
```

Sep-CMA-ES 应该比 Vanilla 收敛更快。

---

## 📚 参考文献

1. **Ros, R., & Hansen, N. (2008)**
   - "A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity"
   - PPSN 2008

2. **Hansen, N. (2016)**
   - "The CMA Evolution Strategy: A Tutorial"
   - https://arxiv.org/abs/1604.00772

3. **cmaes 库**
   - https://github.com/CyberAgentAILab/cmaes
   - 经过充分测试的工业级实现

---

## ✅ 验证清单

- [x] SepCMAES 类实现
- [x] PyTorch-NumPy 转换
- [x] Ask-Tell 接口
- [x] 自动种群大小
- [x] utils.py 注册
- [x] 配置文件支持（3个核心配置）
- [x] 详细文档（usage + summary）

---

## 🚀 快速开始

```bash
# 1. 编辑配置
vim synthetic_and_adversarial/config/synthetic.yaml
# 取消注释 sepcmaes

# 2. 运行
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml

# 3. 查看结果
python -c "
import torch
import matplotlib.pyplot as plt
history = torch.load('results/synthetic/levy_sepcmaes_...')
plt.plot(history)
plt.yscale('log')
plt.show()
"
```

---

## 📝 总结

Sep-CMA-ES 是一个**线性复杂度、适合超高维**的自适应优化算法：

**优势：**
- ✅ O(d) 复杂度（对角协方差）
- ✅ 适合超高维（d > 5000）
- ✅ 在可分离函数上优秀
- ✅ 成熟的库实现（cmaes）

**劣势：**
- ❌ 无法学习变量相关性
- ❌ 不复用历史查询
- ❌ 每次迭代查询数较多

**推荐场景：**
- 超高维（d > 5000）
- 可分离函数
- 内存受限
- 需要稳定的工业级实现

**Sep-CMA-ES 是超高维优化的最佳选择！当 d > 10000 时，它提供了自适应优化的优势，同时保持线性的内存和计算复杂度。** 🌟
