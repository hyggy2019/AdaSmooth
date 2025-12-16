看了你的完整代码和文档，我来全面审查一下实现的正确性：

---

## ✅ **总体评价：实现基本正确**

你已经正确实现了：
- ✅ Vanilla, ZOO, REINFORCE, ZoAR, ZoHS（你自己确认的）
- ✅ TwoPoint（两点式）
- ✅ ES（纯进化策略）
- ✅ xNES
- ✅ SepCMAES

---

## 🔍 **逐个审查**

### **1. ES (纯进化策略)** ✅ 正确

```python
class ES(ZerothOrderOptimizer):
    def estimate_gradient(self, closure):
        noises = []
        fs = []

        for _ in range(self.num_queries):
            noise = self._generate_noise()
            noises.append(noise)
            self._perturb_params(noise, self.mu)
            f_perturbed = closure()
            fs.append(f_perturbed.item())
            self._perturb_params(noise, -self.mu)

        fs = torch.tensor(fs, device=...)

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                for noise, f_val in zip(noises, fs):
                    # ✅ 正确：直接用 f_val，不减基线
                    param.grad += f_val / self.mu * noise[param]

                param.grad.div_(self.num_queries)

        return fs.mean()
```

**数学验证**：
$$\nabla_\theta \approx \frac{1}{n\mu} \sum_{i=1}^{n} F(\theta + \mu\epsilon_i) \cdot \epsilon_i$$

**✅ 完全正确！**

---

### **2. TwoPointMatched (两点式)** ✅ 正确

```python
class TwoPointMatched(ZerothOrderOptimizer):
    def estimate_gradient(self, closure):
        loss = closure()  # baseline f(θ)

        num_directions = self.num_queries // 2

        noises = []
        fs_plus = []
        fs_minus = []

        for _ in range(num_directions):
            noise = self._generate_noise()
            noises.append(noise)

            # f(θ + μu)
            self._perturb_params(noise, self.mu)
            f_plus = closure()
            fs_plus.append(f_plus.item())
            self._perturb_params(noise, -self.mu)

            # f(θ - μu)
            self._perturb_params(noise, -self.mu)
            f_minus = closure()
            fs_minus.append(f_minus.item())
            self._perturb_params(noise, self.mu)

        fs_plus = torch.tensor(fs_plus, device=loss.device)
        fs_minus = torch.tensor(fs_minus, device=loss.device)

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                for noise, f_p, f_m in zip(noises, fs_plus, fs_minus):
                    # ✅ 正确：中心差分公式
                    param.grad += (f_p - f_m) / (2 * self.mu) * noise[param]

                # ✅ 正确：除以方向数
                param.grad.div_(num_directions)

        return loss
```

**数学验证**：
$$\nabla_\theta \approx \frac{1}{m} \sum_{i=1}^{m} \frac{F(\theta + \mu\epsilon_i) - F(\theta - \mu\epsilon_i)}{2\mu} \cdot \epsilon_i$$

其中 $m = \lfloor n/2 \rfloor$

**✅ 完全正确！**

**查询预算匹配**：
- 方向数：`num_queries // 2`
- 每方向查询：2 次（+μ 和 -μ）
- 总查询：1（baseline）+ num_queries

✅ 与 Vanilla 的查询预算完全匹配！

---

### **3. xNES** ⚠️ **有小问题需要修复**

```python
class xNES(ZerothOrderOptimizer):
    def _initialize_xnes(self, param):
        if self.initialized:
            return

        self.dim = param.numel()
        self.sigma_xnes = self.mu  # ❌ 问题在这里
        self.bmat = torch.eye(self.dim, device=param.device, dtype=param.dtype)
        ...
```

**问题**：
```python
self.sigma_xnes = self.mu  # ❌ self.mu 是扰动幅度（通常 0.01）
```

这会导致 `sigma_xnes` 初始值过小（0.01），而 xNES 的 sigma 应该是一个合理的步长（通常 0.1 到 1.0）。

**修复方案**：

#### **方案 A：添加独立的 initial_sigma 参数**（推荐）

```python
class xNES(ZerothOrderOptimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'sgd',
        eta_mu: float = 1.0,
        eta_sigma: float = None,
        eta_bmat: float = None,
        use_fshape: bool = True,
        initial_sigma: float = 0.1,  # ✅ 新增参数
    ):
        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)
        
        self.eta_mu = eta_mu
        self.use_fshape = use_fshape
        self.initial_sigma = initial_sigma  # ✅ 保存
        # ...

    def _initialize_xnes(self, param):
        if self.initialized:
            return

        self.dim = param.numel()
        self.sigma_xnes = self.initial_sigma  # ✅ 使用独立参数
        self.bmat = torch.eye(self.dim, device=param.device, dtype=param.dtype)
        # ...
```

然后在 `utils.py` 中：

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
        initial_sigma=initial_sigma  # ✅ 传入
    )
```

#### **方案 B：直接硬编码为 1.0**（简单但不灵活）

```python
def _initialize_xnes(self, param):
    if self.initialized:
        return

    self.dim = param.numel()
    self.sigma_xnes = 1.0  # ✅ 硬编码合理值
    self.bmat = torch.eye(self.dim, device=param.device, dtype=param.dtype)
    # ...
```

**推荐使用方案 A**，更灵活。

---

### **4. SepCMAES** ✅ **完全正确**

```python
class SepCMAES(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, sigma=0.1, population_size=None):
        # ...
        self.initial_params = np.concatenate(all_params)
        self.dim = len(self.initial_params)

        mean = self.initial_params.copy()
        cma_kwargs = {}
        if population_size is not None:
            cma_kwargs['population_size'] = population_size

        # ✅ 正确使用 cmaes 库
        self.cma_optimizer = SepCMA(mean=mean, sigma=sigma, **cma_kwargs)

    def step(self, closure):
        solutions = []

        # Ask phase
        for _ in range(self.cma_optimizer.population_size):
            x = self.cma_optimizer.ask()
            self._numpy_to_params(x)
            loss = closure()
            solutions.append((x, loss.item()))

        # Tell phase
        self.cma_optimizer.tell(solutions)

        # Update to mean
        self._numpy_to_params(self.cma_optimizer._mean)

        return min(loss for _, loss in solutions)
```

**✅ 实现完全正确！**

- ✅ 正确使用 `cmaes.SepCMA`
- ✅ Ask-Tell 模式正确
- ✅ PyTorch-NumPy 转换正确
- ✅ 独立于 `ZerothOrderOptimizer`（符合你的要求）

---

## 📊 **符号一致性检查**

你提到"注意符号问题"，让我验证一下所有方法的符号：

### **最小化 vs 最大化**

**基类约定**（zo.py 第 76 行）：
```python
param.add_(-lr * param.grad)  # 梯度下降 → 最小化
```

所有方法的符号：

| 方法 | 梯度符号 | 是否正确 |
|------|---------|---------|
| **ES** | `+f_val / mu * noise` | ✅ 正确（最大化→最小化需取负） |
| **Vanilla** | `+(f_plus - f_0) / mu * noise` | ✅ 正确 |
| **TwoPoint** | `+(f_p - f_m) / (2*mu) * noise` | ✅ 正确 |
| **xNES** | `-eta_mu * sigma * grad_direction` | ✅ 正确（已有负号） |
| **SepCMAES** | 直接最小化 `loss` | ✅ 正确（cmaes 默认最小化） |

**✅ 所有符号一致，都是最小化问题！**

---

## ⚠️ **需要修复的问题总结**

### **1. xNES 的 sigma 初始化**

**当前代码**：
```python
self.sigma_xnes = self.mu  # ❌ 太小（0.01）
```

**推荐修复**：
```python
self.sigma_xnes = self.initial_sigma  # ✅ 使用独立参数（默认 0.1）
```

### **2. xNES 必须使用 update_rule='sgd'**

**当前代码**：
```python
update_rule: str = 'radazo',  # ❌ 默认值不对
```

**推荐修复**：
```python
update_rule: str = 'sgd',  # ✅ xNES 必须用 SGD
```

并在 `__init__` 中添加检查：
```python
if update_rule != 'sgd':
    raise ValueError("xNES requires update_rule='sgd'")
```

### **3. xNES 的 lr 应该默认为 1.0**

**当前代码**：
```python
lr: float = 0.001,  # ❌ 会导致有效学习率变成 0.001 × eta_mu
```

**推荐修复**：
```python
lr: float = 1.0,  # ✅ xNES 标准用法
```

---

## 🎯 **最终修复建议**

### **修改 optimizer/zo.py 中的 xNES 类**：

```python
class xNES(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 1.0,  # ✅ 修改：默认 1.0
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'sgd',  # ✅ 修改：默认 sgd
        eta_mu: float = 1.0,
        eta_sigma: float = None,
        eta_bmat: float = None,
        use_fshape: bool = True,
        initial_sigma: float = 0.1,  # ✅ 新增参数
    ):
        # ✅ 新增：检查 update_rule
        if update_rule != 'sgd':
            raise ValueError("xNES requires update_rule='sgd'")
        
        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)
        
        self.eta_mu = eta_mu
        self.use_fshape = use_fshape
        self.initial_sigma = initial_sigma  # ✅ 保存
        
        # ... 其余代码不变

    def _initialize_xnes(self, param):
        if self.initialized:
            return

        self.dim = param.numel()
        self.sigma_xnes = self.initial_sigma  # ✅ 修改：使用 initial_sigma
        self.bmat = torch.eye(self.dim, device=param.device, dtype=param.dtype)
        
        # ... 其余代码不变
```

### **修改 utils.py**：

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
        update_rule='sgd',  # ✅ 强制 SGD
        eta_mu=eta_mu,
        eta_sigma=eta_sigma,
        eta_bmat=eta_bmat,
        use_fshape=use_fshape,
        initial_sigma=initial_sigma  # ✅ 传入
    )
```

---

## ✅ **总结**

### **完全正确的实现**：
1. ✅ ES（纯进化策略）
2. ✅ TwoPointMatched（两点式）
3. ✅ SepCMAES
4. ✅ Vanilla, ZOO, REINFORCE, ZoAR, ZoHS（你已确认）

### **需要小修改的实现**：
5. ⚠️ **xNES**：需要修复 `sigma` 初始化、默认 `update_rule` 和默认 `lr`

修改后，所有实现都将**完全正确**！🎉