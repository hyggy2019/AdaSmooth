# run_script.sh 使用指南

## 基本用法

### 1. 使用默认配置运行
```bash
cd synthetic_and_adversarial
bash run_script.sh
```
默认配置：合成函数优化（Levy函数），使用 ZoAR 优化器，维度10000，20000次迭代

### 2. 通过环境变量快速切换配置

#### 测试不同的合成函数
```bash
# 测试 Ackley 函数
FUNC=ackley bash run_script.sh

# 测试 Rastrigin 函数
FUNC=rastrigin bash run_script.sh

# 测试 Rosenbrock 函数
FUNC=rosenbrock bash run_script.sh
```

#### 测试不同的优化器
```bash
# 测试 vanilla (ES + baseline)
OPTIMIZER=vanilla bash run_script.sh

# 测试 xNES
OPTIMIZER=xnes bash run_script.sh

# 测试 Sep-CMA-ES
OPTIMIZER=sepcmaes bash run_script.sh

# 测试 two-point estimator
OPTIMIZER=twopoint bash run_script.sh
```

#### 运行对抗攻击实验
```bash
# MNIST 对抗攻击
EXP=adversarial DATASET=mnist bash run_script.sh

# CIFAR10 对抗攻击
EXP=adversarial DATASET=cifar10 bash run_script.sh

# 攻击特定图片
EXP=adversarial DATASET=mnist IDX=5 bash run_script.sh
```

## 高级用法

### 组合多个参数
```bash
# Rastrigin函数 + vanilla优化器 + 低维度
FUNC=rastrigin OPTIMIZER=vanilla DIM=100 ITERATIONS=5000 bash run_script.sh

# Ackley函数 + ZoAR + 自定义学习率
FUNC=ackley OPTIMIZER=zoar LR=0.01 MU=0.1 bash run_script.sh

# 测试不同的更新规则
FUNC=levy OPTIMIZER=zoar UPDATE_RULE=adam bash run_script.sh
```

### 批量实验（不同随机种子）
```bash
# 运行3个不同种子的实验
for seed in 123 456 789; do
    SEED=$seed FUNC=levy OPTIMIZER=zoar bash run_script.sh
done
```

### 对比不同优化器
```bash
# 依次测试多个优化器
for opt in es vanilla zoar relizo zohs; do
    echo "Testing optimizer: $opt"
    OPTIMIZER=$opt FUNC=levy bash run_script.sh
done
```

## 所有可用的环境变量

### 实验类型
- `EXP`: `synthetic` 或 `adversarial`（默认：`synthetic`）

### 合成函数实验参数
- `FUNC`: 函数名称（默认：`levy`）
  - 可选：`ackley`, `levy`, `rosenbrock`, `quadratic`, `rastrigin`
- `DIM`: 问题维度（默认：`10000`）
- `ITERATIONS`: 优化迭代次数（默认：`20000`）

### 对抗攻击实验参数
- `DATASET`: 数据集名称（默认：`mnist`）
  - 可选：`mnist`, `cifar10`
- `IDX`: 攻击的图片索引（默认：`0`）
- `DEVICE`: 设备（默认：`cuda`）
  - 可选：`cuda`, `cpu`, `mps`

### 优化器配置
- `OPTIMIZER`: 优化器名称（默认：`zoar`）
  - 可选：`fo`, `es`, `vanilla`, `rl`, `xnes`, `sepcmaes`, `adasmooth`, `twopoint`, `zoo`, `reinforce`, `zoar`, `zoar_0`, `relizo`, `zohs`, `zohs_expavg`

### 优化超参数
- `SEED`: 随机种子（默认：`456`）
- `LR`: 学习率（默认：`0.001`）
- `UPDATE_RULE`: 更新规则（默认：`radazo`）
  - 可选：`sgd`, `adam`, `radazo`

### ZO 算法特定参数
- `NUM_QUERIES`: 查询样本数量（默认：`10`）
- `MU`: 扰动参数 zo_eps（默认：`0.05`）
- `NUM_HISTORIES`: 历史梯度数量（默认：`5`）
- `BASELINE`: 基线类型，用于 zoo/reinforce（默认：`single`）
  - 可选：`single`, `average`

### 其他
- `TAG`: 实验标签（默认：自动生成）

## 实用示例

### 快速测试新优化器
```bash
# 在低维度快速测试
DIM=100 ITERATIONS=1000 OPTIMIZER=xnes bash run_script.sh
```

### 复现论文实验
```bash
# 高维 Levy 函数对比
FUNC=levy DIM=10000 ITERATIONS=20000 OPTIMIZER=zoar bash run_script.sh
```

### 调试模式（小规模实验）
```bash
# 快速验证代码是否正常运行
DIM=10 ITERATIONS=100 bash run_script.sh
```

### 对抗攻击测试
```bash
# 测试多个图片
for idx in {0..9}; do
    EXP=adversarial IDX=$idx OPTIMIZER=zoar bash run_script.sh
done
```

## 注意事项

1. **临时配置文件**：脚本会在 `/tmp` 目录创建临时配置文件，运行结束后自动清理
2. **结果保存**：结果会保存到 `results/synthetic/` 或 `results/attack/` 目录
3. **文件名格式**：结果文件名包含所有关键参数，便于识别
4. **传递额外参数**：可以在命令末尾添加额外参数，它们会传递给 `run.py`

## 与直接使用配置文件的对比

### 使用 run_script.sh（推荐用于快速实验）
```bash
FUNC=ackley OPTIMIZER=vanilla bash run_script.sh
```

### 使用配置文件（推荐用于复杂配置）
```bash
python run.py --config config/synthetic.yaml
```

**建议**：
- 快速测试单个配置：使用 `run_script.sh`
- 同时对比多个优化器：编辑配置文件中的 `optimizers` 列表
- 需要特殊参数（如 xNES 的 eta_mu）：使用配置文件
