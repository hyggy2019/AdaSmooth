# 实验计划与绘图指南 (Experiment Plan & Visualization)

## 1. 待办事项 (To-Do List)* [ ] **Fix Problem**: 修复现有问题。
* [ ] **Ablation Study**: 测试所有 **散度 (Divergence)** 和 **Beta Scheduler** 的组合。
* [ ] **Plotting**: 依照参考代码绘制实验结果图。
* **参考文件**: `/home/zlouyang/ZoAR/figures.ipynb`



## 2. 实验配置 (Experiment Setup)###2.1 算法对比 (Algorithms)* vanilla
* zoar
* relizo
* twopoint
* zohs
* sepcmaes
* adasmooth_es

### 2.2 维度与迭代次数 (Dimensions & Iterations)
| 维度 (Dimension) | 迭代次数 (Iterations) |
| --- | --- |
| 1000 | 10000 |
| 5000 | 15000 |
| 10000 | 20000 |

## 3. 测试任务 (Tasks & Benchmarks)

### 3.1 合成函数 (Synthetic Functions)用于基础性能测试：

* Rosenbrock
* Ackley
* Rastrigin
最终汇报:
1. 收敛曲线 (Convergence Plots)
2. 最终收敛值的Table
3. 每个算法的运行时间。以 vanilla 为基准，计算其他算法的加速比 (speedup)。

### 3.2 对抗攻击 (Adversarial Attacks)用于实际应用场景测试：

* **数据集**:
* MNIST
* CIFAR-10

要求
1. 成功攻击率 (Success Rate)，生成类似这样的表格 （里面的值你需要填充）

| Metric | Vanilla | ZoHS | ZoAR w/o history | ZoAR | SepCMA-ES | AdaSmoothES |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| # Iters ($\times 10^2$) | 23.3 $\pm$ 5.4 | 23.3 $\pm$ 2.6 | 12.4 $\pm$ 1.0 | **8.56 $\pm$ 2.2** |  |  |
| Speedup | 1.0 $\times$ | 1.0 $\times$ | 1.87 $\times$ | **2.72 $\times$** |  |  |
2. 收敛曲线 (Convergence Plots)
3. 每个算法的运行时间。以 vanilla 为基准，计算其他算法的加速比 (speedup)。

## 4. 绘图规范 (Plotting Configuration)在复现 `/home/zlouyang/ZoAR/figures.ipynb` 的图表时，请务必添加以下配置以满足投稿要求（解决 Type 3 字体问题）：

```python
import matplotlib.pyplot as plt
import matplotlib

# =============================================================================
# 推荐的 Matplotlib 投稿配置
# =============================================================================

# 1. 解决 Type 3 字体问题（关键！）
#    告诉 Matplotlib 在 PDF 中使用 Type 42 (TrueType) 字体
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

```