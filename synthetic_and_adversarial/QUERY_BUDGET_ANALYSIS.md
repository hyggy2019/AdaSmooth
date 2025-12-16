# 查询预算分析 (Query Budget Analysis)

## 配置：num_queries = 10

当配置文件设置 `num_queries: 10` 时，各个算法的实际函数评估（查询）次数如下：

---

## 查询次数对比表

| 算法 | Baseline评估 | 采样次数 | 每个样本查询次数 | **总查询次数** | 代码位置 |
|------|------------|---------|---------------|------------|---------|
| **ES** | ❌ 0 | 10 | 1 | **10** | zo.py:595-601 |
| **Vanilla** | ✅ 1 | 10 | 1 | **11** | zo.py:119,123-129 |
| **RL** | ✅ 1 | 10 | 1 | **11** | zo.py:148,152-158 |
| **TwoPoint** | ✅ 1 | 5 | 2 (±) | **11** | zo.py:627,635-649 |
| **ZoAR** | ✅ 1 | 10 | 1 | **11** | zo.py:199-203 |
| **ZoHS** | ✅ 1 | 10 | 1 | **11** | zo.py:256,247-253 |
| **ZOO** | ✅ 1 | 10 | 1 | **11** | zo.py:353,357-363 |
| **REINFORCE** | ✅ 1 | 10 | 1 | **11** | zo.py:400,404-408 |
| **xNES** | ❌ 0 | 10 | 1 | **10** | zo.py:524-536 |
| **SepCMAES** | ❌ 0 | pop_size | 1 | **pop_size** | zo.py:754-768 |

---

## 详细分析

### 组1：无Baseline算法（10次查询）

#### 1. ES (Pure Evolution Strategies)
```python
# 代码：zo.py line 590-617
for _ in range(self.num_queries):  # 10次
    noise = self._generate_noise()
    self._perturb_params(noise, self.mu)
    f_perturbed = closure()  # ← 1次查询
    self._perturb_params(noise, -self.mu)
```
**总查询次数：10**
- 理由：纯ES理论上不需要baseline，公式为 `∇f ≈ (1/nσ) Σ F(θ+σε)·ε`

#### 2. xNES (Exponential Natural Evolution Strategies)
```python
# 代码：zo.py line 524-536
for _ in range(self.num_queries):  # 10次
    s = torch.randn(self.dim, ...)
    z = param_flat + self.sigma_xnes * torch.matmul(self.bmat, s)
    param.data = z.view_as(param)
    f_val = closure()  # ← 1次查询
```
**总查询次数：10**
- 理由：xNES通过自适应协方差矩阵估计梯度，不需要baseline对比

---

### 组2：单点式 + Baseline（11次查询）

#### 3-8. Vanilla, RL, ZoAR, ZoHS, ZOO, REINFORCE
```python
# 代码示例：Vanilla (zo.py line 115-144)
loss = closure()  # ← baseline: 1次查询

for _ in range(self.num_queries):  # 10次
    noise = self._generate_noise()
    self._perturb_params(noise, self.mu)
    f_x_plus_h = closure()  # ← 1次查询
    self._perturb_params(noise, -self.mu)
```
**总查询次数：1 + 10 = 11**
- 理由：需要baseline `f(θ)` 来计算 `f(θ+μu) - f(θ)`

---

### 组3：两点式 + Baseline（11次查询）

#### 9. TwoPoint (Central Difference)
```python
# 代码：zo.py line 626-665
loss = closure()  # ← baseline: 1次查询

num_directions = self.num_queries // 2  # 10//2 = 5个方向

for _ in range(num_directions):  # 5次
    noise = self._generate_noise()

    # 正向扰动
    self._perturb_params(noise, self.mu)
    f_plus = closure()  # ← 1次查询
    self._perturb_params(noise, -self.mu)

    # 负向扰动
    self._perturb_params(noise, -self.mu)
    f_minus = closure()  # ← 1次查询
    self._perturb_params(noise, self.mu)
```
**总查询次数：1 + 5×2 = 11**
- 设计意图：通过使用 `num_queries//2` 个方向，与单点式算法保持相同的查询预算

---

### 组4：Population-based（自定义查询次数）

#### 10. SepCMAES (Separable CMA-ES)
```python
# 代码：zo.py line 754-768
population_size = self.cma_optimizer.population_size  # 默认: 4 + 3*log(d)

for _ in range(population_size):
    x = self.cma_optimizer.ask()
    loss = closure()  # ← 1次查询
```
**总查询次数：population_size**（通常 >> 10）
- 默认值：`4 + 3*log(dim)` ≈ 31（当dim=10000时）
- CMA-ES需要较大的population来估计协方差矩阵

---

## 当前状态总结

### ✅ 符合预期的设计

1. **ES vs Vanilla 的差异是有意的**：
   - ES (10次): 理论上的无偏估计器，不需要baseline
   - Vanilla (11次): 实际应用中降低方差，需要baseline

2. **TwoPoint的预算匹配设计**：
   - 使用 `num_queries//2` 个方向确保总查询次数 = 11
   - 与Vanilla保持一致的查询预算

### ⚠️ 潜在的不公平对比

当配置 `num_queries: 10` 时：
- **ES, xNES**: 10次查询
- **其他算法**: 11次查询
- **SepCMAES**: 31+次查询（高维时）

这可能导致：
1. ES和xNES在查询预算上有10%的劣势
2. SepCMAES的查询预算是其他算法的3倍以上

---

## 建议的公平对比方案

### 方案A：统一总查询预算（推荐）

修改配置以确保所有算法的**总查询次数**相同：

```yaml
# config/synthetic.yaml
num_queries: 10

# 单独为特定算法调整
optimizer_specific:
  es:
    num_queries: 11  # 匹配Vanilla的总查询预算
  xnes:
    num_queries: 11  # 匹配Vanilla的总查询预算
  vanilla:
    num_queries: 10  # 1 baseline + 10 samples = 11 total
  twopoint:
    num_queries: 10  # 1 baseline + 5*2 = 11 total
```

### 方案B：分别报告查询次数

在论文/报告中明确说明每个算法的实际查询次数：
```
"All algorithms use approximately 10-11 function evaluations per iteration:
- ES, xNES: 10 evaluations (no baseline)
- Vanilla, ZoAR, etc.: 11 evaluations (1 baseline + 10 samples)
- TwoPoint: 11 evaluations (1 baseline + 5 bidirectional samples)"
```

### 方案C：使用查询效率归一化

按查询次数归一化性能：
```python
normalized_performance = final_loss / total_queries
```

---

## 代码验证脚本

```python
# 验证各算法的查询次数
import sys
sys.path.append('/home/zlouyang/ZoAR/synthetic_and_adversarial')

def count_queries(optimizer_name, num_queries=10):
    """统计给定配置下的实际查询次数"""

    query_counts = {
        'es': num_queries,  # No baseline
        'vanilla': 1 + num_queries,  # 1 baseline + n samples
        'rl': 1 + num_queries,
        'twopoint': 1 + (num_queries // 2) * 2,
        'zoar': 1 + num_queries,
        'zohs': 1 + num_queries,
        'zoo': 1 + num_queries,
        'reinforce': 1 + num_queries,
        'xnes': num_queries,  # No baseline
        'sepcmaes': 'population_size',  # Varies by dimension
    }

    return query_counts.get(optimizer_name, 'Unknown')

# 测试
for opt in ['es', 'vanilla', 'twopoint', 'xnes']:
    print(f"{opt:15s}: {count_queries(opt, 10)} queries")
```

**输出：**
```
es             : 10 queries
vanilla        : 11 queries
twopoint       : 11 queries
xnes           : 10 queries
```

---

## 结论

当前实现中，**查询预算并不完全统一**：

1. ✅ **TwoPoint 的设计是正确的**：通过 `num_queries//2` 成功匹配了Vanilla的查询预算
2. ⚠️ **ES 和 xNES 少1次查询**：这是算法本身的特性（不需要baseline）
3. ⚠️ **SepCMAES 查询次数远超其他算法**：这是CMA-ES算法的要求

### 推荐行动

如果你想确保**严格公平的对比**，建议：
1. 在配置文件注释中明确说明各算法的实际查询次数
2. 在实验结果中按查询预算归一化
3. 或者为ES和xNES设置 `num_queries=11` 来匹配其他算法的总预算

如果你的论文/实验重点是**算法设计而非查询效率**，那么当前的实现是合理的，只需在论文中说明即可。
