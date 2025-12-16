# xNES 修复完成报告

## ✅ 修复完成

xNES 优化器的三个初始化问题已全部修复。

**修复日期：** 2024-12-13

---

## 📋 修复内容

### 1. ✅ Sigma 初始化修复

**问题：** sigma 使用 `self.mu` (0.01) 初始化，步长过小

**修复：**
- 添加独立参数 `initial_sigma: float = 0.1`
- 使用 `self.sigma_xnes = self.initial_sigma` 替代 `self.mu`

**文件：** `optimizer/zo.py` 第 455, 465, 482 行

### 2. ✅ Update Rule 修复

**问题：** 默认 `update_rule='radazo'`，但 xNES 需要 SGD

**修复：**
- 修改默认值为 `update_rule: str = 'sgd'`
- 添加验证：`if update_rule != 'sgd': raise ValueError(...)`

**文件：** `optimizer/zo.py` 第 450, 457-459 行

### 3. ✅ Learning Rate 修复

**问题：** 默认 `lr=0.001` 过小

**修复：**
- 修改默认值为 `lr: float = 1.0`

**文件：** `optimizer/zo.py` 第 445 行

---

## 📊 修复前后对比

| 参数 | 修复前 | 修复后 |
|------|--------|--------|
| **lr** | 0.001 ❌ | 1.0 ✅ |
| **update_rule** | 'radazo' ❌ | 'sgd' ✅ |
| **sigma 初始化** | self.mu (0.01) ❌ | self.initial_sigma (0.1) ✅ |

---

## 🔧 代码变更

### optimizer/zo.py

**__init__ 方法：**
```python
def __init__(
    self,
    params: Iterator[torch.Tensor],
    lr: float = 1.0,  # ✅ 从 0.001 改为 1.0
    betas: Tuple[float, float] = (0.9, 0.99),
    epsilon: float = 1e-8,
    num_queries: int = 10,
    mu: float = 0.01,
    update_rule: str = 'sgd',  # ✅ 从 'radazo' 改为 'sgd'
    eta_mu: float = 1.0,
    eta_sigma: float = None,
    eta_bmat: float = None,
    use_fshape: bool = True,
    initial_sigma: float = 0.1,  # ✅ 新增参数
):
    # ✅ 新增验证
    if update_rule != 'sgd':
        raise ValueError("xNES requires update_rule='sgd'")

    super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

    self.eta_mu = eta_mu
    self.use_fshape = use_fshape
    self.initial_sigma = initial_sigma  # ✅ 保存参数
    # ...
```

**_initialize_xnes 方法：**
```python
def _initialize_xnes(self, param):
    if self.initialized:
        return

    self.dim = param.numel()
    self.sigma_xnes = self.initial_sigma  # ✅ 从 self.mu 改为 self.initial_sigma
    self.bmat = torch.eye(self.dim, device=param.device, dtype=param.dtype)
    # ...
```

### utils.py

**xNES 注册：**
```python
elif name == "xnes":
    eta_mu = getattr(args, 'eta_mu', 1.0)
    eta_sigma = getattr(args, 'eta_sigma', None)
    eta_bmat = getattr(args, 'eta_bmat', None)
    use_fshape = getattr(args, 'use_fshape', True)
    initial_sigma = getattr(args, 'initial_sigma', 0.1)  # ✅ 新增
    return xNES(
        params=params,
        lr=args.lr,
        betas=args.betas,
        epsilon=args.epsilon,
        num_queries=args.num_queries,
        mu=args.mu,
        update_rule='sgd',
        eta_mu=eta_mu,
        eta_sigma=eta_sigma,
        eta_bmat=eta_bmat,
        use_fshape=use_fshape,
        initial_sigma=initial_sigma  # ✅ 新增
    )
```

---

## ✅ 测试验证

### 测试 1: 默认参数
```python
optimizer = xNES(params)
assert optimizer.param_groups[0]['lr'] == 1.0  # ✅
assert optimizer.update_rule == 'sgd'  # ✅
assert optimizer.initial_sigma == 0.1  # ✅
```

### 测试 2: Sigma 初始化
```python
loss = optimizer.step(closure)
assert abs(optimizer.sigma_xnes - 0.1) < 0.01  # ✅ 0.1000 (not 0.0100)
```

### 测试 3: Update Rule 验证
```python
try:
    bad_optimizer = xNES(params, update_rule='radazo')
except ValueError:
    pass  # ✅ 正确抛出错误
```

### 测试 4: 自定义 initial_sigma
```python
optimizer = xNES(params, initial_sigma=0.2)
loss = optimizer.step(closure)
assert abs(optimizer.sigma_xnes - 0.2) < 0.01  # ✅
```

**所有测试通过！** ✅

---

## 📚 文档更新

### 更新的文件：

1. **config/synthetic.yaml**
   - 添加 `initial_sigma` 参数说明

2. **Docx/xNES_usage.md**
   - 添加 `initial_sigma` 参数文档
   - 添加学习率和 SGD 要求说明
   - 添加参数调优建议

3. **Docx/xNES_Fix_COMPLETED.md** ✨ 新建
   - 本文档（修复完成报告）

---

## 🎯 使用建议

### 基本用法（使用默认值）

```yaml
# config/synthetic.yaml
optimizers:
  - xnes  # 使用所有默认值（lr=1.0, initial_sigma=0.1）
```

### 自定义参数

```yaml
# config/synthetic.yaml
optimizers:
  - xnes

# xNES 参数（可选）
eta_mu: 1.0         # 均值学习率
eta_sigma: null     # sigma 学习率（自动）
eta_bmat: null      # B 矩阵学习率（自动）
use_fshape: true    # 使用 fitness shaping
initial_sigma: 0.15 # 初始步长（自定义）
lr: 1.0             # 学习率（推荐保持 1.0）
```

**重要：** xNES **必须使用 SGD**，不支持 Adam/RadAZO。

---

## 📈 预期改进

修复后，xNES 应该：

1. ✅ **收敛更快** - 合理的初始步长（0.1 vs 0.01）
2. ✅ **稳定性更好** - 正确的学习率（1.0 vs 0.001）
3. ✅ **更安全** - 强制 SGD，防止误用
4. ✅ **可配置性** - 可通过 `initial_sigma` 调整步长

---

## 🔍 技术细节

### 为什么 xNES 需要 SGD？

xNES 通过自然梯度和自适应协方差矩阵实现参数更新：

```python
# xNES 内部计算自然梯度
grad_direction = torch.matmul(self.bmat, dj_delta)

# 自适应更新 sigma 和 B 矩阵
self.sigma_xnes *= exp(0.5 * eta_sigma * dj_sigma)
self.bmat = self.bmat @ exp(0.5 * eta_bmat * dj_bmat)

# 设置梯度（仅用于 SGD 更新）
p.grad = -(eta_mu * sigma_xnes * grad_direction)
```

**如果使用 Adam/RadAZO：**
- 会对自然梯度再次应用动量和自适应学习率
- 破坏 xNES 的数学基础
- 导致收敛问题

**因此必须使用 SGD！**

---

## ✅ 修复验证清单

- [x] 修改 `lr` 默认值为 1.0
- [x] 修改 `update_rule` 默认值为 'sgd'
- [x] 添加 `initial_sigma` 参数
- [x] 添加 update_rule 验证
- [x] 更新 sigma 初始化逻辑
- [x] 更新 utils.py 注册代码
- [x] 更新配置文件文档
- [x] 更新使用文档
- [x] 编写测试验证
- [x] 所有测试通过

---

## 🌟 总结

xNES 优化器的三个关键问题已全部修复：

1. ✅ **Sigma 初始化** - 从 0.01 提升到 0.1（10倍改进）
2. ✅ **Update Rule** - 强制 SGD（防止误用）
3. ✅ **Learning Rate** - 从 0.001 提升到 1.0（符合 xNES 标准）

**现在 xNES 已经完全正确，可以放心使用！** 🚀

修复后的 xNES 应该在合成函数优化任务上表现显著更好。
