# AdaSmoothES v2: 模块化散度与温度调度

## 概述

AdaSmoothES v2 是 AdaSmoothES 的模块化扩展版本，支持：

1. **多种散度函数** (Divergence Functionals)
2. **多种温度调度策略** (Temperature Schedules)
3. **灵活组合** 以获得最佳性能

## 架构

```
optimizer/
├── divergences.py              # 散度函数模块
│   ├── KLDivergence            # KL 散度 (默认)
│   ├── ReverseKLDivergence     # 反向 KL 散度
│   ├── ChiSquaredDivergence    # χ² 散度
│   ├── RenyiDivergence         # Rényi 散度
│   ├── TsallisDivergence       # Tsallis 散度
│   └── HuberDivergence         # Huber 散度
│
├── temperature_schedules.py    # 温度调度模块
│   ├── ConstantSchedule        # 恒定温度
│   ├── LinearSchedule          # 线性衰减
│   ├── ExponentialSchedule     # 指数衰减
│   ├── PolynomialSchedule      # 多项式衰减 (默认)
│   ├── CosineAnnealingSchedule # 余弦退火
│   ├── StepSchedule            # 阶梯衰减
│   ├── AdaptiveSchedule        # 自适应温度
│   └── CyclicSchedule          # 周期性温度
│
└── adasmooth_es_v2.py          # 主优化器 (使用上述模块)
```

## 散度函数详解

### 1. KL Divergence (默认)

```
π*(x) ∝ π_ref(x) · exp(-β·F(x))
w_k = softmax(-β·(f_k - b))
```

**特性**:
- Mode-seeking: 偏好覆盖低函数值的模式
- Boltzmann 分布
- RL 和 ES 中最常用

**适用场景**: 一般用途，默认选择

### 2. Reverse KL Divergence

```
π*(x) ∝ π_ref(x) / (1 + β·F(x))
w_k ∝ 1 / (1 + β·(f_k - b))
```

**特性**:
- Mean-seeking: 匹配参考分布的均值
- 对异常值更鲁棒

**适用场景**: 有噪声或异常值的问题

### 3. χ² Divergence

```
π*(x) ∝ π_ref(x) · (1 - β·F(x))
w_k ∝ max(0, 1 - β·(f_k - b))
```

**特性**:
- 二次惩罚
- 对大偏差更鲁棒

**适用场景**: 希望避免过度集中的权重

### 4. Rényi Divergence

```
π*(x) ∝ π_ref(x) · [exp(-β·F(x))]^(1/α)
w_k ∝ exp(-β/α · (f_k - b))
```

**参数**: `alpha` (α > 0, α ≠ 1)
- α → 1: 接近 KL 散度
- α = 2: 接近 χ² 散度
- α < 1: 更具探索性
- α > 1: 更具开发性

**适用场景**: 需要调整探索-开发平衡

### 5. Tsallis Divergence

```
π*(x) ∝ π_ref(x) · [1 - (1-q)·β·F(x)]^(1/(1-q))
```

**参数**: `q` (q > 0, q ≠ 1)
- q → 1: 接近 KL 散度
- q < 1: 重尾分布
- q > 1: 轻尾分布

**适用场景**: 需要非广延熵的问题

### 6. Huber Divergence

```
|A_k| ≤ δ: w_k ∝ exp(-β·A_k)  (KL 区域)
|A_k| > δ: w_k ∝ exp(-β·δ·sign(A_k))  (饱和)
```

**参数**: `delta` (阈值)

**特性**:
- 对异常值鲁棒
- 结合 KL 和二次的优点

**适用场景**: 有大异常值的问题

## 温度调度详解

### 1. Polynomial Schedule (默认)

```
β_t = β_0 / (1 + λ·t)^p
```

**参数**:
- `beta_init`: 初始温度 (默认 10.0)
- `decay_rate`: 衰减率 λ (默认 0.001)
- `power`: 幂次 p (默认 1.0)

**特性**:
- p=1: 双曲衰减
- p>1: 更快衰减
- p<1: 更慢衰减

### 2. Exponential Schedule

```
β_t = β_0 · exp(-λ·t)
```

**特性**: 初期快速衰减，后期缓慢

### 3. Cosine Annealing Schedule

```
β_t = β_min + 0.5·(β_0 - β_min)·(1 + cos(π·t/T))
```

**特性**: S 型平滑曲线，深度学习中常用

### 4. Linear Schedule

```
β_t = β_0 - decay·t
```

**特性**: 最简单的退火策略

### 5. Step Schedule

```
β_t = β_0 · γ^(floor(t / step_size))
```

**参数**:
- `step_size`: 每隔多少迭代降温
- `gamma`: 衰减因子

**特性**: 分段常数，突然降温

### 6. Adaptive Schedule

**特性**:
- 卡住时增加 β (更多探索)
- 改进时降低 β (更多开发)
- 自动调整

### 7. Cyclic Schedule

```
β_t 在 β_min 和 β_max 之间周期振荡
```

**模式**: triangular, sine, saw

**特性**: 周期性探索/开发，可能逃出局部最优

### 8. Constant Schedule

```
β_t = β_0
```

**特性**: 固定探索率

## 使用方法

### 方法 1: 配置文件

```yaml
# config/synthetic.yaml

optimizers:
  - adasmooth_es_v2

# 散度配置
divergence: kl  # 'kl', 'reverse_kl', 'chi2', 'renyi', 'tsallis', 'huber'

# Rényi 散度参数 (如果使用 renyi)
renyi_alpha: 2.0

# Tsallis 散度参数 (如果使用 tsallis)
tsallis_q: 2.0

# Huber 散度参数 (如果使用 huber)
huber_delta: 1.0

# 温度调度配置
temperature_schedule: polynomial  # 'constant', 'linear', 'exponential', 'polynomial', 'cosine', 'step', 'cyclic', 'adaptive'
beta_init: 10.0
beta_decay: 0.001  # 用于 polynomial, exponential
beta_min: 0.01     # 最小温度
poly_power: 1.0    # 用于 polynomial

# 或者使用 cosine
# temperature_schedule: cosine
# beta_init: 10.0
# beta_min: 0.1
# (num_iterations 会自动从配置获取)

# 或者使用 cyclic
# temperature_schedule: cyclic
# beta_min: 5.0
# beta_max: 15.0
# cycle_length: 500
# cyclic_mode: triangular  # 'triangular', 'sine', 'saw'
```

然后运行:
```bash
python run.py --config config/synthetic.yaml
```

### 方法 2: Python 直接使用

```python
import torch
from optimizer.adasmooth_es_v2 import AdaSmoothESv2

param = torch.randn(1000)

# 示例 1: KL + Polynomial (默认)
optimizer = AdaSmoothESv2(
    params=[param],
    num_queries=24
)

# 示例 2: Rényi + Cosine
optimizer = AdaSmoothESv2(
    params=[param],
    num_queries=24,
    divergence='renyi',
    divergence_kwargs={'alpha': 2.0},
    temperature_schedule='cosine',
    temperature_kwargs={'beta_init': 10.0, 'beta_min': 0.1, 'total_iterations': 10000}
)

# 示例 3: Chi-squared + Adaptive
optimizer = AdaSmoothESv2(
    params=[param],
    num_queries=24,
    divergence='chi2',
    temperature_schedule='adaptive',
    temperature_kwargs={'beta_init': 10.0, 'beta_min': 0.1, 'beta_max': 100.0}
)

# 示例 4: Huber + Cyclic
optimizer = AdaSmoothESv2(
    params=[param],
    num_queries=24,
    divergence='huber',
    divergence_kwargs={'delta': 1.0},
    temperature_schedule='cyclic',
    temperature_kwargs={'beta_min': 5.0, 'beta_max': 15.0, 'cycle_length': 500}
)

# 优化循环
for iteration in range(10000):
    loss = optimizer.step(closure)
```

## 测试脚本

我们提供了系统化的测试脚本来比较不同配置：

```bash
# 测试所有散度 (固定 polynomial 温度)
python test_divergences_schedules.py --test divergences

# 测试所有温度调度 (固定 KL 散度)
python test_divergences_schedules.py --test schedules

# 测试最佳组合
python test_divergences_schedules.py --test combinations

# 测试全部
python test_divergences_schedules.py --test all

# 自定义参数
python test_divergences_schedules.py --test all --func ackley --dim 500 --iters 5000 --queries 20
```

## 推荐配置

### 平滑函数 (Rosenbrock, Sphere)

**配置 1: KL + Polynomial (默认)**
```yaml
divergence: kl
temperature_schedule: polynomial
beta_init: 10.0
beta_decay: 0.001
```

**配置 2: Chi² + Cosine**
```yaml
divergence: chi2
temperature_schedule: cosine
beta_init: 10.0
beta_min: 0.1
```

### 多峰函数 (Ackley, Levy, Rastrigin)

**配置 1: Cyclic 温度 (逃出局部最优)**
```yaml
divergence: kl
temperature_schedule: cyclic
beta_min: 5.0
beta_max: 20.0
cycle_length: 500
cyclic_mode: triangular
```

**配置 2: Rényi 散度 (更具探索性)**
```yaml
divergence: renyi
renyi_alpha: 0.5  # < 1 更具探索性
temperature_schedule: polynomial
beta_init: 15.0
beta_decay: 0.0005  # 更慢衰减
```

### 噪声问题

**配置: Huber + Adaptive**
```yaml
divergence: huber
huber_delta: 1.0
temperature_schedule: adaptive
beta_init: 10.0
beta_min: 0.1
beta_max: 50.0
```

### 不确定问题类型

**配置: Adaptive 温度**
```yaml
divergence: kl
temperature_schedule: adaptive
beta_init: 10.0
beta_min: 0.1
beta_max: 100.0
```

## 实验建议

1. **从默认配置开始**: KL + Polynomial
2. **如果收敛慢**: 尝试更快的温度衰减 (增大 `beta_decay`)
3. **如果陷入局部最优**: 尝试 Cyclic 或 Adaptive 温度
4. **如果有噪声**: 尝试 Huber 或 Reverse KL 散度
5. **如果需要更多探索**: 尝试 Rényi (α < 1) 或增大 `beta_init`

## 获取运行时信息

```python
# 获取当前配置信息
info = optimizer.get_info()
print(info)
# {
#     'divergence': 'KL',
#     'temperature_schedule': 'Polynomial(β0=10.0, λ=0.001, p=1.0)',
#     'current_beta': 8.52,
#     'current_sigma': 0.048,
#     'iteration': 1000,
#     'num_queries': 24,
#     'dimension': 1000,
#     'baseline_type': 'mean'
# }

# 访问历史
history = optimizer.history
betas = history['beta']  # 温度历史
weights = history['weights']  # 权重历史
losses = history['f_values']  # 函数值历史
```

## 理论背景

所有散度都求解优化问题:

```
min E_π[F(x)] + (1/β) * D(π || π_ref)
```

其中:
- `F(x)`: 目标函数
- `D`: 散度函数
- `β`: 温度参数 (控制探索-开发平衡)
- `π`: 搜索分布
- `π_ref`: 参考分布

不同的散度 `D` 导致不同的最优策略 `π*` 和权重计算方式。

详细理论见: `/home/zlouyang/ZoAR/Docx/AdaSepCMA.md`

## 性能对比

基于初步测试 (Rosenbrock, d=1000, K=24, 10k iters):

| 配置 | 最终损失 | 时间 |
|------|---------|------|
| KL + Polynomial | 1099 | 22s |
| Chi² + Polynomial | ~1100 | 22s |
| Rényi(α=2) + Polynomial | ~1100 | 22s |

更多详细测试待完成...

## 总结

AdaSmoothES v2 提供了灵活的模块化框架来探索不同的散度和温度策略。根据问题特性选择合适的配置可以显著提升性能。
