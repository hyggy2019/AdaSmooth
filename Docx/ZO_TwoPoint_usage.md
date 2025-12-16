# 两点式（Two-Point）ZO梯度估计器使用说明

## 原理对比

### 单点式（One-Point / Forward Difference）
```
∇f(θ) ≈ (f(θ + μu) - f(θ)) / μ
```
- 使用前向差分
- 每个方向需要 1 次额外查询（+ 1 次基线查询）
- 查询总数：`1 + num_queries`

### 两点式（Two-Point / Central Difference）
```
∇f(θ) ≈ (f(θ + μu) - f(θ - μu)) / (2μ)
```
- 使用中心差分
- 每个方向需要 2 次查询（+ μu 和 - μu）
- 理论上比单点式更精确（二阶近似 vs 一阶近似）
- 为匹配查询预算，使用 `num_queries//2` 个方向
- 查询总数：`1 + num_queries` （与单点式相同）

## 实现细节

已在 `synthetic_and_adversarial/optimizer/zo.py` 中实现 `TwoPointMatched` 类：

```python
class TwoPointMatched(ZerothOrderOptimizer):
    def estimate_gradient(self, closure):
        loss = closure()  # baseline f(θ)
        num_directions = self.num_queries // 2

        for each direction:
            f_plus = f(θ + μu)   # forward query
            f_minus = f(θ - μu)  # backward query
            grad += (f_plus - f_minus) / (2μ) * u

        grad /= num_directions
```

## 已创建的配置文件

### 1. 通用合成函数对比
`config/synthetic-twopoint.yaml`
- 对比 vanilla（单点式）、twopoint（两点式）、zoar

### 2. Rastrigin 函数专用
`config/rastrigin-twopoint.yaml`
- 在高度多峰的 Rastrigin 函数上测试两点式效果
- 包含详细的注释说明

### 3. 对抗攻击场景
`config/adversarial-twopoint.yaml`
- 在黑盒对抗攻击任务上测试两点式

## 运行示例

### 在合成函数上对比单点式 vs 两点式

```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic-twopoint.yaml
```

### 在 Rastrigin 函数上测试

```bash
cd synthetic_and_adversarial
python run.py --config config/rastrigin-twopoint.yaml
```

### 在对抗攻击上测试

```bash
cd synthetic_and_adversarial
python run.py --config config/adversarial-twopoint.yaml
```

## 预期优势

两点式方法在以下情况下可能表现更好：

1. **函数平滑性高**：中心差分对平滑函数的梯度估计更准确
2. **噪声环境**：对称采样可以部分抵消噪声
3. **精度优先**：当查询成本不是主要瓶颈时

## 查询预算匹配

配置示例：`num_queries: 10`

- **Vanilla（单点式）**:
  - 方向数：10
  - 每方向查询：1 次（+ μu）
  - 总查询：1（baseline）+ 10 = 11

- **TwoPoint（两点式）**:
  - 方向数：10 // 2 = 5
  - 每方向查询：2 次（+ μu 和 - μu）
  - 总查询：1（baseline）+ 10 = 11

两种方法使用相同的查询预算，可以公平对比。

## 结果文件

结果保存格式：
```
results/synthetic/{func_name}_twopoint_{update_rule}_d{dim}_ni{iterations}_lr{lr}_nq{queries}_mu{mu}_nh{histories}_s{seed}.pt
```

## 在其他配置中使用

在任何配置文件的 `optimizers` 列表中添加 `twopoint`：

```yaml
optimizers:
  - vanilla    # 单点式基线
  - twopoint   # 两点式（本实现）
  - zoar       # ZoAR 方法
  - relizo     # ReLIZO 方法
```

## 技术细节

- 继承自 `ZerothOrderOptimizer` 基类
- 支持所有更新规则：SGD, Adam, RadAZO
- 与其他优化器共享相同的参数接口
- 自动在 `utils.py` 的 `get_optimizer()` 中注册
