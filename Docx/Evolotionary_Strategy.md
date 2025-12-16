好的，我用数学公式详细解释这些算法的差别：

## **核心梯度估计公式对比**

### **1. Vanilla (标准ES)**
论文中的Algorithm 1实现：
$$\nabla_\theta F(\theta) \approx \frac{1}{n\sigma} \sum_{i=1}^{n} F(\theta + \sigma\epsilon_i) \cdot \epsilon_i$$

代码实现：
```python
grad += (f(θ + μϵ) - f(θ)) / μ · ϵ
grad /= n
```

**数学形式**：
$$\nabla_\theta \approx \frac{1}{n\mu} \sum_{i=1}^{n} [F(\theta + \mu\epsilon_i) - F(\theta)] \cdot \epsilon_i$$

---

### **2. ZOO (Zeroth-Order Optimizer)**

**baseline="single"模式**（与Vanilla相同）：
$$\nabla_\theta \approx \frac{1}{n\mu} \sum_{i=1}^{n} [F(\theta + \mu\epsilon_i) - F(\theta)] \cdot \epsilon_i$$

**baseline="average"模式**：
$$\nabla_\theta \approx \frac{1}{n\mu} \sum_{i=1}^{n} \left[F(\theta + \mu\epsilon_i) - \frac{1}{n}\sum_{j=1}^{n}F(\theta + \mu\epsilon_j)\right] \cdot \epsilon_i$$

**关键差异**：用样本均值 $\bar{F} = \frac{1}{n}\sum_{j} F(\theta + \mu\epsilon_j)$ 替代 $F(\theta)$

---

### **3. REINFORCE**

与ZOO完全相同，只是命名体现强化学习思想：

**baseline="single"**：
$$\nabla_\theta \approx \frac{1}{n\mu} \sum_{i=1}^{n} [R_i - R_0] \cdot \epsilon_i$$
其中 $R_i = F(\theta + \mu\epsilon_i)$，$R_0 = F(\theta)$

**baseline="average"**：
$$\nabla_\theta \approx \frac{1}{n\mu} \sum_{i=1}^{n} [R_i - \bar{R}] \cdot \epsilon_i$$
其中 $\bar{R} = \frac{1}{n}\sum_{j=1}^{n} R_j$

---

### **4. Reinforcement_Learning (排序变换)**

这是论文Section 2.1提到的**fitness shaping**：

**步骤1：排序变换**
$$\text{rank}(R_i) = \text{argsort}(R_1, ..., R_n)[i]$$
$$\tilde{R}_i = 2 \cdot \text{rank}(R_i)$$

**步骤2：中心化**
$$R'_i = \tilde{R}_i - \frac{1}{n}\sum_{j=1}^{n}\tilde{R}_j$$

**步骤3：梯度估计**
$$\nabla_\theta \approx \frac{1}{n\mu} \sum_{i=1}^{n} R'_i \cdot \epsilon_i$$

**效果**：
- 最差的样本权重 → 0
- 最好的样本权重 → $2n$
- 对outlier不敏感

---

### **5. TwoPointMatched (双点有限差分)**

**标准二阶有限差分**：
$$\nabla_\theta F(\theta) \approx \frac{F(\theta + \mu\epsilon) - F(\theta - \mu\epsilon)}{2\mu}$$

**向量化形式**：
$$\nabla_\theta \approx \frac{1}{m} \sum_{i=1}^{m} \frac{F(\theta + \mu\epsilon_i) - F(\theta - \mu\epsilon_i)}{2\mu} \cdot \epsilon_i$$

其中 $m = \lfloor n/2 \rfloor$（匹配查询预算）

代码实现：
```python
num_directions = num_queries // 2
for each direction:
    f_plus = F(θ + μϵ)   # 正向扰动
    f_minus = F(θ - μϵ)  # 负向扰动
    grad += (f_plus - f_minus) / (2μ) · ϵ
grad /= num_directions
```

---

## **数学对比总结表**

| 算法 | 梯度估计公式 | 基线类型 | 采样方式 |
|------|-------------|---------|---------|
| **Vanilla** | $\frac{1}{n\mu}\sum [F(\theta+\mu\epsilon) - F(\theta)] \epsilon$ | 当前点 $F(\theta)$ | 单侧 |
| **ZOO (single)** | $\frac{1}{n\mu}\sum [F(\theta+\mu\epsilon) - F(\theta)] \epsilon$ | 当前点 $F(\theta)$ | 单侧 |
| **ZOO (average)** | $\frac{1}{n\mu}\sum [F(\theta+\mu\epsilon) - \bar{F}] \epsilon$ | 样本均值 $\bar{F}$ | 单侧 |
| **RL (rank)** | $\frac{1}{n\mu}\sum R'_i \epsilon_i$ | 排序后中心化 | 单侧 |
| **TwoPoint** | $\frac{1}{m}\sum \frac{F(\theta+\mu\epsilon) - F(\theta-\mu\epsilon)}{2\mu} \epsilon$ | 对称差分 | 双侧 |

---

## **方差分析**

**理论方差顺序**（从高到低）：
$$\text{Var}[\text{Vanilla}] > \text{Var}[\text{ZOO-avg}] > \text{Var}[\text{RL-rank}] > \text{Var}[\text{TwoPoint}]$$

**原因**：
- **Vanilla/ZOO-single**：$F(\theta)$ 可能远离扰动样本的均值
- **ZOO-average**：$\bar{F}$ 更接近样本中心，减少方差
- **RL-rank**：排序变换消除了异常值影响
- **TwoPoint**：对称采样 $\pm\epsilon$ 天然消除一阶误差项

抱歉我漏掉了！让我补充 **ES** 的定义。

在这个代码库中，**没有单独的 `ES` 类**，但从论文来看，"ES" (Evolution Strategies) 是一个**总称**，指代所有这些方法。

不过，如果你是问论文中提到的**标准ES算法**，那它对应的是：

---

## **ES (Evolution Strategies) - 论文标准算法**

### **论文Algorithm 1的数学形式**

$$\nabla_\theta E_{\epsilon \sim \mathcal{N}(0,I)} F(\theta + \sigma\epsilon) = \frac{1}{\sigma} E_{\epsilon}[F(\theta + \sigma\epsilon) \cdot \epsilon]$$

**蒙特卡洛估计**：
$$\nabla_\theta \approx \frac{1}{n\sigma} \sum_{i=1}^{n} F(\theta + \sigma\epsilon_i) \cdot \epsilon_i$$

**关键特征**：
- **没有减去基线** $F(\theta)$
- 直接使用 $F(\theta + \sigma\epsilon_i)$ 的值
- 这是**最原始的自然进化策略 (NES)** 形式

---

## **与代码中其他方法的对比**

### **数学公式对比**

| 方法 | 梯度估计公式 | 是否减基线 |
|------|-------------|-----------|
| **论文ES** | $\frac{1}{n\sigma}\sum F(\theta+\sigma\epsilon) \cdot \epsilon$ | ❌ 否 |
| **Vanilla** | $\frac{1}{n\mu}\sum [F(\theta+\mu\epsilon) - F(\theta)] \cdot \epsilon$ | ✅ 是 |
| **ZOO/REINFORCE** | $\frac{1}{n\mu}\sum [F(\theta+\mu\epsilon) - \text{baseline}] \cdot \epsilon$ | ✅ 是 |

---

## **为什么代码中的实现都减去了基线？**

这是**方差缩减技术**！

### **数学推导**

原始ES梯度：
$$\nabla_\theta = E_{\epsilon}[F(\theta + \sigma\epsilon) \cdot \frac{\epsilon}{\sigma}]$$

添加基线 $b$（任意常数）：
$$\nabla_\theta = E_{\epsilon}[(F(\theta + \sigma\epsilon) - b) \cdot \frac{\epsilon}{\sigma}]$$

**为什么可以这样做？**

因为：
$$E_{\epsilon}[b \cdot \epsilon] = b \cdot E_{\epsilon}[\epsilon] = b \cdot 0 = 0$$

所以**梯度的期望不变**，但**方差显著降低**！

### **最优基线选择**

理论上最优基线是：
$$b^* = \frac{E[F(\theta+\sigma\epsilon) \cdot \|\epsilon\|^2]}{E[\|\epsilon\|^2]}$$

实践中常用的近似：
- $b = F(\theta)$ （Vanilla）
- $b = \frac{1}{n}\sum F(\theta + \sigma\epsilon_i)$ （ZOO-average）
- $b = \text{rank-normalized}$ （RL-rank）

---

## **完整对比表**

| 算法 | 数学公式 | 基线 | 方差 | 来源 |
|------|---------|------|------|------|
| **论文原始ES** | $\frac{1}{n\sigma}\sum F_i \epsilon_i$ | 无 | 最高 | 论文Algorithm 1（理论形式） |
| **Vanilla (ES+baseline)** | $\frac{1}{n\mu}\sum (F_i - F_0) \epsilon_i$ | $F_0 = F(\theta)$ | 高 | 实际实现 |
| **ZOO-average** | $\frac{1}{n\mu}\sum (F_i - \bar{F}) \epsilon_i$ | $\bar{F} = \text{mean}(F_i)$ | 中 | 方差缩减 |
| **RL-rank** | $\frac{1}{n\mu}\sum R'_i \epsilon_i$ | 排序中心化 | 低 | 论文Section 2.1 fitness shaping |
| **TwoPoint** | $\frac{1}{m}\sum \frac{F_+ - F_-}{2\mu} \epsilon_i$ | 对称差分 | 最低 | 二阶有限差分 |

其中 $F_i = F(\theta + \mu\epsilon_i)$，$F_0 = F(\theta)$

---

## **论文中实际使用的ES**

论文的实验中提到：

> "We also find it useful to perform **fitness shaping** by applying a rank transformation to the returns before computing each parameter update."

所以论文实际使用的是：

$$\nabla_\theta \approx \frac{1}{n\sigma} \sum_{i=1}^{n} \text{rank\_normalized}(F_i) \cdot \epsilon_i$$

这对应代码中的 **`Reinforcement_Learning` 类**！

---

## **总结**

- **纯ES**：$\frac{1}{n\sigma}\sum F_i \epsilon_i$ （理论，高方差）
- **实用ES** = ES + baseline + fitness shaping
- 代码中的 `Vanilla`/`ZOO`/`REINFORCE` 都是**ES的变体**
- **`Reinforcement_Learning`** 最接近论文实际使用的算法