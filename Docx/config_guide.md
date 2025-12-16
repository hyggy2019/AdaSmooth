# 配置文件使用指南

## 📁 配置文件结构

本项目只有 **3 个核心配置文件**，所有功能通过这3个文件使用：

```
synthetic_and_adversarial/config/
├── synthetic.yaml          # 合成函数测试（主要配置）
├── synthetic-baseline.yaml # 基线方法对比
└── adversarial.yaml        # 黑盒对抗攻击
```

---

## 🎯 配置文件说明

### 1. `synthetic.yaml` - 合成函数测试 ⭐ 推荐

**用途：** 最常用的配置，支持所有合成函数和优化器

**默认设置：**
```yaml
func_name: levy        # 测试函数
optimizers:
  - vanilla            # ES + 基线
  - zoar_0             # ZoAR (无历史)
  - zoar               # ZoAR (带历史)
  - relizo             # ReLIZO
  - zohs               # ZOHS
```

**支持的函数：**
- `ackley` - Ackley 函数
- `levy` - Levy 函数
- `rosenbrock` - Rosenbrock 函数
- `quadratic` - 二次函数
- `rastrigin` - Rastrigin 函数（高度多峰）

**支持的优化器（通过注释切换）：**
```yaml
optimizers:
  # - fo        # 真实梯度
  # - es        # 纯ES（无基线）
  - vanilla     # ES + 单点基线 ✅ 默认启用
  # - rl        # ES + fitness shaping
  # - twopoint  # 两点式（中心差分）
  - zoar_0      # ZoAR (无历史) ✅ 默认启用
  - zoar        # ZoAR (带历史) ✅ 默认启用
  - relizo      # ReLIZO ✅ 默认启用
  - zohs        # ZOHS ✅ 默认启用
  # - zohs_expavg
```

---

### 2. `synthetic-baseline.yaml` - 基线方法对比

**用途：** 对比不同基线策略（ZOO, REINFORCE）

**特殊参数：**
```yaml
baseline: average  # "single" 或 "average"
```

**默认优化器：**
```yaml
optimizers:
  - vanilla    # 标准单点基线
  - twopoint   # 两点式
  - zoar       # ZoAR
  - relizo     # ReLIZO
```

**可选优化器（注释中）：**
```yaml
# - es         # 纯ES
# - zoo        # ZOO（需要 baseline 参数）
# - reinforce  # REINFORCE（需要 baseline 参数）
```

---

### 3. `adversarial.yaml` - 黑盒对抗攻击

**用途：** MNIST/CIFAR10 对抗攻击

**特殊参数：**
```yaml
dataset: mnist  # "mnist" 或 "cifar10"
idx: 1          # 攻击的图像索引
device: cpu     # "cuda", "cpu", "mps"
```

**默认优化器：**
```yaml
optimizers:
  - vanilla
  - zoar_0
  - zoar
  - relizo
  - zohs
```

---

## 🛠️ 使用示例

### 例1：测试 Rastrigin 函数

**方法1：** 直接修改配置文件

编辑 `config/synthetic.yaml`：
```yaml
func_name: rastrigin  # 从 levy 改为 rastrigin
```

运行：
```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml
```

---

### 例2：对比 ES 方法

编辑 `config/synthetic.yaml`：
```yaml
optimizers:
  - es       # 纯ES（取消注释）
  - vanilla  # ES + 基线
  - rl       # ES + 排序（取消注释）
  - zoar     # ZoAR
```

运行：
```bash
python run.py --config config/synthetic.yaml
```

---

### 例3：单点式 vs 两点式

编辑 `config/synthetic.yaml`：
```yaml
func_name: rastrigin  # 改为 rastrigin

optimizers:
  - vanilla   # 单点式（前向差分）
  - twopoint  # 两点式（中心差分，取消注释）
  - zoar      # ZoAR
```

运行：
```bash
python run.py --config config/synthetic.yaml
```

---

### 例4：使用基线方法

直接运行（使用默认配置）：
```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic-baseline.yaml
```

或修改 `baseline` 参数：
```yaml
baseline: single  # 改为 "single"（使用 F(θ) 作为基线）
```

---

### 例5：对抗攻击（CIFAR10）

编辑 `config/adversarial.yaml`：
```yaml
dataset: cifar10  # 从 mnist 改为 cifar10
x_dim: 3072       # CIFAR10 图像大小 (32×32×3)
```

运行：
```bash
python run.py --config config/adversarial.yaml
```

---

## 🎨 优化器选择指南

### 按方差排序（从低到高）

```
twopoint < zoar < zohs < relizo < rl < vanilla < es
```

### 按查询效率排序

```
zoar (复用) > es (无基线) > vanilla/twopoint (标准)
```

### 场景推荐

**平滑凸函数：**
```yaml
optimizers:
  - vanilla
  - twopoint  # 最低方差
  - zoar
```

**高度多峰（Rastrigin）：**
```yaml
optimizers:
  - rl        # fitness shaping
  - zoar      # 历史平滑
  - relizo    # 自适应复用
```

**查询受限：**
```yaml
optimizers:
  - zoar      # 最高效率
  - es        # 无额外查询（但方差高）
```

**理论研究：**
```yaml
optimizers:
  - es        # 理论基线
  - vanilla   # ES + 基线
  - rl        # ES + 排序
```

---

## 📊 参数调优建议

### 学习率 (lr)

```yaml
# 合成函数
lr: 0.001  # 默认值

# 对抗攻击
lr: 0.01   # 通常更高
```

### 查询数量 (num_queries)

```yaml
# 合成函数（高维）
num_queries: 10

# 对抗攻击（低维，查询昂贵）
num_queries: 2

# twopoint 实际使用：num_queries//2 个方向
```

### 扰动系数 (mu)

```yaml
# 合成函数
mu: 0.05

# 对抗攻击
mu: 0.5  # 更大的扰动
```

### 历史数量 (num_histories)

```yaml
# ZoAR, ZOHS 使用
num_histories: 5   # 默认值
num_histories: 0   # zoar_0（无历史）
num_histories: 15  # 更多历史（更平滑，但可能过时）
```

---

## 🔍 结果分析

### 结果文件位置

```
results/
├── synthetic/     # 合成函数结果
└── attack/        # 对抗攻击结果
```

### 文件名格式

```
{func}_{opt}_{rule}_d{dim}_ni{iter}_lr{lr}_nq{nq}_mu{mu}_nh{nh}_s{seed}.pt
```

**示例：**
```
levy_vanilla_radazo_d10000_ni20000_lr0.001_nq10_mu0.05_nh5_s456.pt
```

### 加载和分析

```python
import torch
import matplotlib.pyplot as plt

# 加载结果
history = torch.load('results/synthetic/levy_vanilla_radazo_...')

# 绘图
plt.plot(history)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Optimization History')
plt.show()

# 统计
print(f"Final value: {history[-1]}")
print(f"Best value: {min(history)}")
print(f"Improvement: {history[0] / history[-1]:.2f}x")
```

---

## 💡 最佳实践

### 1. 从默认配置开始

先运行默认配置，理解基本行为：
```bash
python run.py --config config/synthetic.yaml
```

### 2. 一次改一个变量

对比实验时，只修改一个参数：
```yaml
# 实验1：默认
func_name: levy
optimizers: [vanilla, zoar]

# 实验2：只改函数
func_name: rastrigin  # 只改这个
optimizers: [vanilla, zoar]  # 保持不变
```

### 3. 使用注释管理优化器

通过注释快速切换：
```yaml
optimizers:
  # - es        # 实验1：测试纯ES
  - vanilla     # 实验1：基线
  # - twopoint  # 实验2：测试两点式
  - zoar        # 所有实验都用
```

### 4. 记录实验设置

在运行前记录配置：
```bash
# 复制配置（可选）
cp config/synthetic.yaml config/my_experiment.yaml

# 运行并记录
python run.py --config config/my_experiment.yaml | tee experiment.log
```

---

## ❓ 常见问题

### Q1: 如何添加新优化器？

取消注释即可：
```yaml
optimizers:
  - vanilla
  - twopoint  # 取消 # 即可启用
```

### Q2: 如何测试所有函数？

多次运行，每次改 `func_name`：
```bash
# 测试 Ackley
sed -i 's/func_name: levy/func_name: ackley/' config/synthetic.yaml
python run.py --config config/synthetic.yaml

# 测试 Rastrigin
sed -i 's/func_name: ackley/func_name: rastrigin/' config/synthetic.yaml
python run.py --config config/synthetic.yaml
```

### Q3: baseline 参数什么时候需要？

只有 `zoo` 和 `reinforce` 需要：
```yaml
optimizers:
  - zoo       # 需要 baseline 参数
  - reinforce # 需要 baseline 参数
  - vanilla   # 不需要
```

---

## 📚 更多信息

详细文档：
- `Docx/quick_reference.md` - 快速参考
- `Docx/ES_usage.md` - ES 方法详解
- `Docx/ZO_TwoPoint_usage.md` - TwoPoint 详解
- `Docx/Rastrigin_usage.md` - Rastrigin 函数
- `CLAUDE.md` - 项目总览
