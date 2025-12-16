# 📋 实验运行指南 (Experiment Guide)

根据 `/home/zlouyang/ZoAR/TODO.md` 的要求配置的完整实验方案。

---

## 🎯 实验配置总览

### 1. 合成函数 (Synthetic Functions)

**函数**: Rosenbrock, Ackley, Rastrigin

**维度与迭代次数**:
| 维度 (d) | 迭代次数 | 配置文件 |
|---------|---------|----------|
| 1000 | 10000 | `config/synthetic.yaml` |
| 5000 | 15000 | `config/synthetic-d5000.yaml` |
| 10000 | 20000 | `config/synthetic-d10000.yaml` |

**测试算法**:
- vanilla (基准，用于计算加速比)
- zoar
- relizo
- twopoint
- zohs
- sepcmaes
- adasmooth_es (最优配置: CMA Match + Decay)

**输出**:
1. ✅ 收敛曲线 (Convergence Plots)
2. ✅ 最终收敛值表格 (Final Loss Table)
3. ✅ 运行时间和加速比 (Runtime & Speedup vs Vanilla)

---

### 2. 对抗攻击 (Adversarial Attacks)

**数据集**: MNIST, CIFAR-10

**配置文件**:
- MNIST: `config/adversarial.yaml`
- CIFAR-10: `config/adversarial-cifar10.yaml`

**测试算法**: 同上（vanilla, zoar, relizo, twopoint, zohs, sepcmaes, adasmooth_es）

**输出**:
1. ✅ 成功攻击率表格 (Success Rate & Speedup Table)
2. ✅ 收敛曲线 (Convergence Plots)
3. ✅ 运行时间和加速比 (Runtime & Speedup vs Vanilla)

---

## 🚀 运行实验

### 方法1: 运行单个实验

使用 `run_script_simple.sh` 运行单个配置：

```bash
cd synthetic_and_adversarial

# 方法1: 直接修改 run_script_simple.sh 中的配置路径
bash run_script_simple.sh

# 方法2: 使用 run.py 直接指定
python run.py --config config/synthetic.yaml
```

### 方法2: 运行特定实验

使用 `run_all_todo.sh` 运行特定实验：

```bash
cd synthetic_and_adversarial

# 合成函数: rosenbrock, d=1000
bash run_all_todo.sh rosenbrock 1000 synthetic

# 合成函数: ackley, d=5000
bash run_all_todo.sh ackley 5000 synthetic

# 合成函数: rastrigin, d=10000
bash run_all_todo.sh rastrigin 10000 synthetic

# 对抗攻击: MNIST
bash run_all_todo.sh mnist 1000 adversarial

# 对抗攻击: CIFAR-10
bash run_all_todo.sh cifar10 1000 adversarial
```

### 方法3: 自动运行所有实验

运行 TODO.md 中的所有实验：

```bash
cd synthetic_and_adversarial
bash run_all_experiments.sh
```

这将自动运行：
- ✅ 3个函数 × 3个维度 = 9个合成函数实验
- ✅ 2个数据集 = 2个对抗攻击实验
- **总计**: 11个实验

---

## 📊 生成图表和表格

实验完成后，运行绘图脚本：

```bash
cd synthetic_and_adversarial
python plot_all_results.py
```

这将生成：

### 合成函数输出
1. **收敛曲线**: `figures/<func>_d<dim>_convergence.pdf`
   - 例如: `figures/rosenbrock_d1000_convergence.pdf`

2. **最终收敛值表格**:
   - CSV: `figures/synthetic_final_losses.csv`
   - LaTeX: `figures/synthetic_final_losses.tex`

3. **运行时间和加速比**:
   - CSV: `figures/synthetic_speedup.csv`
   - LaTeX: `figures/synthetic_speedup.tex`

### 对抗攻击输出
1. **收敛曲线**: `figures/<dataset>_adversarial_convergence.pdf`
   - 例如: `figures/mnist_adversarial_convergence.pdf`

2. **成功攻击率和加速比表格**:
   - CSV: `figures/adversarial_metrics.csv`
   - LaTeX: `figures/adversarial_metrics.tex`
   - 格式按照 TODO.md 要求

---

## 📂 配置文件详解

### 合成函数配置 (`config/synthetic.yaml`)

```yaml
# 基本配置
func_name: rosenbrock  # 函数名
dimension: 1000        # 维度
num_iterations: 10000  # 迭代次数
seed: 456             # 随机种子

# 测试的算法
optimizers:
  - vanilla           # 基准
  - zoar
  - relizo
  - twopoint
  - zohs
  - sepcmaes
  - adasmooth_es     # 最优配置

# ZO参数
num_queries: 10       # K=10
mu: 0.05             # 扰动参数
num_histories: 5     # 历史梯度数

# AdaSmoothES最优配置
adaptive_beta: cma_match  # CMA Match调度器
cma_decay: 0.001          # 时间衰减
baseline: mean            # 方差缩减
```

**要修改函数**: 改变 `func_name` (rosenbrock, ackley, rastrigin)

**要修改维度**: 使用对应的配置文件
- d=1000: `config/synthetic.yaml`
- d=5000: `config/synthetic-d5000.yaml`
- d=10000: `config/synthetic-d10000.yaml`

---

### 对抗攻击配置 (`config/adversarial.yaml`)

```yaml
# 基本配置
dataset: mnist        # 数据集
model: cnn           # 攻击模型
num_iterations: 3000 # 迭代次数
seed: 456           # 随机种子

# 测试的算法
optimizers:
  - vanilla
  - zoar
  - relizo
  - twopoint
  - zohs
  - sepcmaes
  - adasmooth_es

# ZO参数（与合成函数一致）
num_queries: 10
mu: 0.05
num_histories: 5

# AdaSmoothES最优配置
adaptive_beta: cma_match
cma_decay: 0.001
baseline: mean
```

**要修改数据集**:
- MNIST: `config/adversarial.yaml`
- CIFAR-10: `config/adversarial-cifar10.yaml`

---

## ✅ 验证配置正确性

运行一个快速测试：

```bash
cd synthetic_and_adversarial

# 测试 AdaSmoothES 最优配置 (应该得到 ~986.65)
python run.py --config config/synthetic.yaml
```

**预期输出**:
```
adasmooth_es optimized value: 986.65, Time taken: 12.49 seconds
```

如果结果接近986.65，说明配置正确！✅

---

## 📊 TODO.md 要求对照

| 要求 | 实现 | 文件 |
|------|------|------|
| **算法**: vanilla, zoar, relizo, twopoint, zohs, sepcmaes, adasmooth_es | ✅ | 所有配置文件 |
| **维度**: 1000, 5000, 10000 | ✅ | `synthetic.yaml`, `synthetic-d5000.yaml`, `synthetic-d10000.yaml` |
| **迭代**: 10000, 15000, 20000 | ✅ | 对应配置文件 |
| **函数**: Rosenbrock, Ackley, Rastrigin | ✅ | 通过 `--func_name` 或修改配置 |
| **数据集**: MNIST, CIFAR-10 | ✅ | `adversarial.yaml`, `adversarial-cifar10.yaml` |
| **收敛曲线** | ✅ | `plot_all_results.py` |
| **最终收敛值表格** | ✅ | `plot_all_results.py` → CSV & LaTeX |
| **运行时间和加速比** | ✅ | 自动记录和计算 |
| **成功攻击率表格** | ✅ | `plot_all_results.py` → CSV & LaTeX |
| **Type 42字体** | ✅ | `matplotlib.rcParams['pdf.fonttype'] = 42` |

---

## 🎯 AdaSmoothES 最优配置

所有配置文件已更新为使用 **AdaSmoothES 最优方案**：

```yaml
adaptive_beta: cma_match  # CMA Match调度器
cma_decay: 0.001          # 时间衰减率
baseline: mean            # 均值baseline
```

**性能** (K=10, Rosenbrock d=1000):
- Loss: **986.65** 🏆
- vs SepCMAES: **-9.6%**
- vs Fixed Scheduler: **-12.1%**

---

## 📝 结果目录结构

```
synthetic_and_adversarial/
├── config/                      # 配置文件
│   ├── synthetic.yaml          # d=1000
│   ├── synthetic-d5000.yaml    # d=5000
│   ├── synthetic-d10000.yaml   # d=10000
│   ├── adversarial.yaml        # MNIST
│   └── adversarial-cifar10.yaml # CIFAR-10
│
├── results/                     # 实验结果
│   ├── synthetic/              # 合成函数结果
│   │   └── *.pt               # 优化历史
│   └── attack/                 # 对抗攻击结果
│       └── *.pt               # 优化历史
│
├── figures/                     # 图表输出
│   ├── *_convergence.pdf      # 收敛曲线
│   ├── synthetic_final_losses.csv     # 表格
│   ├── synthetic_final_losses.tex
│   ├── synthetic_speedup.csv
│   ├── synthetic_speedup.tex
│   ├── adversarial_metrics.csv
│   └── adversarial_metrics.tex
│
├── run_script_simple.sh        # 简单运行脚本
├── run_all_todo.sh            # 单个实验运行
├── run_all_experiments.sh     # 全部实验自动运行
└── plot_all_results.py        # 绘图脚本
```

---

## 🔧 故障排查

### 问题1: "No such file or directory"
**解决**: 确保在 `synthetic_and_adversarial/` 目录下运行

### 问题2: "CUDA out of memory"
**解决**: 修改配置文件中的 `device: cuda` 为 `device: cpu`

### 问题3: 结果与预期不符
**解决**:
1. 检查 `seed: 456` 是否一致
2. 检查 `num_queries: 10` 是否正确
3. 检查 AdaSmoothES 配置是否完整

### 问题4: sepcmaes 运行错误
**解决**: 确保 `population_size: 10` 与 `num_queries: 10` 一致

---

## 📧 快速开始示例

```bash
# 1. 进入目录
cd /home/zlouyang/ZoAR/synthetic_and_adversarial

# 2. 运行一个测试（验证配置）
python run.py --config config/synthetic.yaml

# 3. 运行所有实验（自动化）
bash run_all_experiments.sh

# 4. 生成所有图表
python plot_all_results.py

# 完成！检查 figures/ 目录
ls figures/
```

---

**配置完成时间**: 2025-12-16
**配置状态**: ✅ 所有配置已就绪
**AdaSmoothES**: ✅ 使用最优方案 (CMA Match + Decay)
**准备运行**: ✅ 可直接使用 `bash run_all_experiments.sh`
