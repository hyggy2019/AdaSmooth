# Evolution Strategies (ES) 使用说明

## ES 方法族对比

### 1. 纯ES（Pure ES）- 理论形式

**数学公式：**
```
∇f(θ) ≈ (1/nσ) Σ F(θ + σεi) · εi
```

**特点：**
- ❌ **不减去基线**
- 📈 **方差最高**（理论形式）
- 🎓 对应论文 Algorithm 1 的原始公式

**代码实现：**
```python
class ES(ZerothOrderOptimizer):
    def estimate_gradient(self, closure):
        for each direction εi:
            f_val = F(θ + σεi)
            grad += f_val / σ · εi
        grad /= n
```

---

### 2. Vanilla - ES + 单点基线

**数学公式：**
```
∇f(θ) ≈ (1/nμ) Σ [F(θ + μεi) - F(θ)] · εi
```

**特点：**
- ✅ 减去基线 `F(θ)`
- 📉 方差降低（与纯ES相比）
- 🔧 最常用的实用版本

---

### 3. RL (Reinforcement_Learning) - ES + Fitness Shaping

**数学公式：**
```
Step 1: rank(Ri) = argsort(R1, ..., Rn)
Step 2: R̃i = 2 · rank(Ri)
Step 3: R'i = R̃i - mean(R̃)
Step 4: ∇f ≈ (1/nμ) Σ R'i · εi
```

**特点：**
- ✅ 排序变换（fitness shaping）
- 📉 对异常值不敏感
- 🎯 论文实际使用的方法

---

### 4. ZOO/REINFORCE - ES + 可配置基线

**两种基线模式：**

**baseline="single"**（与 Vanilla 相同）:
```
∇f ≈ (1/nμ) Σ [F(θ+με) - F(θ)] · ε
```

**baseline="average"**（样本均值）:
```
∇f ≈ (1/nμ) Σ [F(θ+με) - F̄] · ε
其中 F̄ = (1/n) Σ F(θ+μεj)
```

---

### 5. ZoAR - ES + 查询复用 + 历史基线

**特点：**
- ✅ 复用历史查询
- ✅ 使用历史梯度作为基线
- 📉 方差最低（查询预算相同情况下）

---

## 方差对比（从高到低）

```
ES (纯) > Vanilla > ZOO-average > RL (rank) > ZoAR
```

---

## 实现文件

### 已添加的 ES 优化器

**文件：** `synthetic_and_adversarial/optimizer/zo.py`

```python
class ES(ZerothOrderOptimizer):
    """
    Pure Evolution Strategies - Original formulation from paper Algorithm 1.

    Gradient estimator without baseline subtraction:
        ∇f(θ) ≈ (1/nσ) Σ F(θ + σεi) · εi
    """
```

**特点：**
- 不需要 baseline 参数
- 直接使用 F(θ+σε) 的值
- 返回采样值的均值作为 loss 估计

---

## 配置文件

### 1. ES 方法对比
**文件：** `config/es-comparison.yaml`

```yaml
optimizers:
  - es       # 纯ES（无基线）
  - vanilla  # ES + 单点基线
  - rl       # ES + fitness shaping
  - zoar     # ES + 查询复用
```

**运行：**
```bash
cd synthetic_and_adversarial
python run.py --config config/es-comparison.yaml
```

### 2. Rastrigin + ES
**文件：** `config/rastrigin-es.yaml`

在高度多峰的 Rastrigin 函数上测试 ES 方法：

```bash
cd synthetic_and_adversarial
python run.py --config config/rastrigin-es.yaml
```

### 3. 综合基线测试
**文件：** `config/synthetic-baseline.yaml` ✨ 已更新

包含所有优化器的综合配置，ES 默认注释（可取消注释启用）：

```yaml
optimizers:
  # - es         # 取消注释以启用纯ES
  - vanilla      # ES + 基线
  - twopoint     # 两点式
  - zoar         # ZoAR
  - relizo       # ReLIZO
```

---

## 使用示例

### 启用纯ES

在任何配置文件的 `optimizers` 列表中添加：

```yaml
optimizers:
  - es  # 纯ES（无基线）
```

### 对比 ES vs Vanilla

```yaml
optimizers:
  - es       # 无基线（高方差）
  - vanilla  # 有基线（低方差）
```

### 对比所有 ES 变体

```yaml
optimizers:
  - es       # 纯ES
  - vanilla  # ES + 基线
  - rl       # ES + 排序
  - zoar     # ES + 查询复用
```

---

## 理论背景

### 为什么可以减去基线？

**数学推导：**

原始ES梯度：
```
∇θ = E[F(θ + σε) · ε/σ]
```

添加基线 b（任意常数）：
```
∇θ = E[(F(θ + σε) - b) · ε/σ]
```

因为 `E[b · ε] = b · E[ε] = 0`，所以：
- ✅ 梯度期望不变
- 📉 方差显著降低

### 最优基线

理论最优基线：
```
b* = E[F(θ+σε) · ||ε||²] / E[||ε||²]
```

实践中的近似：
- `b = F(θ)` - Vanilla 使用
- `b = mean(F(θ+σε))` - ZOO-average 使用
- `b = rank-normalized` - RL 使用

---

## 查询成本

所有ES变体的查询成本：

| 方法 | 每次迭代查询数 |
|------|---------------|
| ES (纯) | n |
| Vanilla | 1 + n |
| RL | 1 + n |
| ZOO | 1 + n |
| ZoAR | n（复用历史） |

其中 n = `num_queries`

---

## 预期性能

在不同场景下的推荐：

### 平滑凸函数
- 推荐：Vanilla, TwoPoint
- ES (纯) 方差太高，不推荐

### 高度多峰（如 Rastrigin）
- 推荐：RL (fitness shaping), ZoAR
- ES (纯) 可能陷入局部最优

### 查询成本受限
- 推荐：ZoAR（查询复用）
- 避免：ES (纯)

### 理论研究 / 基准对比
- ES (纯) 可作为理论基线

---

## 结果分析

结果保存在 `results/synthetic/`，文件名格式：
```
{func_name}_{optimizer}_radazo_d{dim}_ni{iter}_lr{lr}_nq{queries}_mu{mu}_nh{hist}_s{seed}.pt
```

加载并对比：
```python
import torch
import matplotlib.pyplot as plt

es_history = torch.load('results/synthetic/levy_es_radazo_...')
vanilla_history = torch.load('results/synthetic/levy_vanilla_radazo_...')

plt.plot(es_history, label='ES (pure)')
plt.plot(vanilla_history, label='Vanilla (ES+baseline)')
plt.legend()
plt.yscale('log')
plt.show()
```

预期：Vanilla 收敛更稳定，方差更小。
