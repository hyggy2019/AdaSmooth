# Rastrigin函数使用说明

## 已完成的修改

### 1. 函数实现
在 `synthetic_and_adversarial/model/synthetic_functions.py` 中添加了 `Rastrigin` 类：

```python
class Rastrigin(SyntheticFunction):
    def forward(self) -> torch.Tensor:
        x = self.x
        A = 10
        d = self.dim
        return A * d + torch.sum(x ** 2 - A * torch.cos(2 * torch.pi * x))
```

**数学公式**: f(x) = 10n + Σ(x_i² - 10·cos(2πx_i))

**性质**:
- 全局最小值: f(0) = 0 在 x = 0 处
- 多峰函数，有大量局部最小值
- 测试域: x_i ∈ [-5.12, 5.12]

### 2. 配置文件

创建了专用配置文件 `synthetic_and_adversarial/config/rastrigin.yaml`

### 3. 更新现有配置

更新了 `synthetic_and_adversarial/config/synthetic.yaml` 的注释，添加 "rastrigin" 选项

### 4. 更新文档

更新了 `CLAUDE.md`，在所有相关位置添加了Rastrigin函数的说明

## 如何运行

### 使用专用配置文件运行Rastrigin优化：

```bash
cd synthetic_and_adversarial
python run.py --config config/rastrigin.yaml
```

### 修改现有配置文件运行：

编辑 `config/synthetic.yaml`，将 `func_name` 改为 `rastrigin`：

```yaml
func_name: rastrigin  # 改为rastrigin
```

然后运行：

```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml
```

## 配置参数说明

关键参数（在 `config/rastrigin.yaml` 中）：

- `dimension: 10000` - 问题维度
- `num_iterations: 20000` - 优化迭代次数
- `lr: 0.001` - 学习率
- `num_queries: 10` - ZO梯度估计的查询样本数
- `mu: 0.05` - 扰动参数（zo_eps）
- `num_histories: 5` - ZoAR使用的历史梯度数量

## 预期结果

- 结果保存在 `results/synthetic/` 目录
- 文件名格式: `rastrigin_{optimizer}_{update_rule}_d{dim}_ni{iterations}_lr{lr}_nq{queries}_mu{mu}_nh{histories}_s{seed}.pt`
- 可以使用 `torch.load()` 加载结果查看优化历史

## 验证实现

函数已正确添加到 `get_synthetic_funcs()` 字典中，可通过以下方式验证：

```python
from model.synthetic_functions import get_synthetic_funcs
import torch

# 在全局最小值点测试（应该接近0）
x_init = torch.zeros(100)
func = get_synthetic_funcs('rastrigin', x_init)
result = func()
print(f"Rastrigin at x=0: {result.item()}")  # 应该 ≈ 0
```

## 对比实验

配置文件中已设置多个优化器进行对比：
- `vanilla` - 标准ZO方法
- `zoar_0` - ZoAR无历史版本
- `zoar` - 完整ZoAR方法
- `relizo` - ReLIZO方法
- `zohs` - ZOHS方法

所有优化器会在同一设置下运行，便于性能对比。
