# 🚀 快速开始 - TODO.md 实验

## ✅ 已完成配置

所有实验配置已按照 `/home/zlouyang/ZoAR/TODO.md` 要求完成！

---

## 📁 配置文件列表

### 合成函数
- ✅ `config/synthetic.yaml` - Rosenbrock d=1000, 10000 iters
- ✅ `config/synthetic-d5000.yaml` - d=5000, 15000 iters
- ✅ `config/synthetic-d10000.yaml` - d=10000, 20000 iters

### 对抗攻击
- ✅ `config/adversarial.yaml` - MNIST, 3000 iters
- ✅ `config/adversarial-cifar10.yaml` - CIFAR-10, 3000 iters

---

## 🎯 算法列表（所有配置）

所有配置文件已包含以下算法：
1. vanilla (基准，用于加速比计算)
2. zoar
3. relizo
4. twopoint
5. zohs
6. sepcmaes
7. adasmooth_es (最优配置)

---

## 🏆 AdaSmoothES 最优配置

所有配置已应用最优方案：

```yaml
adaptive_beta: cma_match
cma_decay: 0.001
baseline: mean
```

**性能**: 986.65 (K=10, Rosenbrock d=1000) 🏆

---

## 🚀 运行方式

### 方式1: 单个实验

```bash
# 修改 run_script_simple.sh 中的配置路径
bash run_script_simple.sh

# 或直接用 python
python run.py --config config/synthetic.yaml
```

### 方式2: 指定实验

```bash
# 语法: bash run_all_todo.sh <function> <dimension> <type>

# 示例
bash run_all_todo.sh rosenbrock 1000 synthetic
bash run_all_todo.sh ackley 5000 synthetic
bash run_all_todo.sh mnist 1000 adversarial
```

### 方式3: 全自动运行

```bash
# 运行所有11个实验（3函数×3维度 + 2数据集）
bash run_all_experiments.sh
```

---

## 📊 要求对照表

| TODO.md 要求 | 状态 | 说明 |
|-------------|------|------|
| **算法**: vanilla, zoar, relizo, twopoint, zohs, sepcmaes, adasmooth_es | ✅ | 所有配置包含 |
| **维度**: 1000, 5000, 10000 | ✅ | 3个配置文件 |
| **迭代**: 10000, 15000, 20000 | ✅ | 按维度配置 |
| **函数**: Rosenbrock, Ackley, Rastrigin | ✅ | 通过参数指定 |
| **数据集**: MNIST, CIFAR-10 | ✅ | 2个配置文件 |
| **收敛曲线** | ✅ | plot_all_results.py |
| **最终收敛值表格** | ✅ | CSV + LaTeX |
| **运行时间和加速比** | ✅ | 自动计算 |
| **Type 42字体** | ✅ | matplotlib配置 |

---

## 📈 预期输出

### 合成函数
1. 收敛曲线: `figures/<func>_d<dim>_convergence.pdf`
2. 最终Loss表格: `figures/synthetic_final_losses.csv`
3. 加速比表格: `figures/synthetic_speedup.csv`

### 对抗攻击
1. 收敛曲线: `figures/<dataset>_adversarial_convergence.pdf`
2. 攻击成功率表格: `figures/adversarial_metrics.csv`
3. 加速比: 以 vanilla 为基准

---

## ✅ 验证

运行快速测试验证配置：

```bash
python run.py --config config/synthetic.yaml
```

**预期输出**:
```
adasmooth_es optimized value: 986.65, Time taken: 12.49 seconds
```

如果结果接近 986.65，配置正确！✅

---

## 📖 详细文档

查看完整指南: `/home/zlouyang/ZoAR/EXPERIMENT_GUIDE.md`

---

**配置完成**: ✅ 2025-12-16
**准备运行**: ✅ 使用 `bash run_all_experiments.sh`
