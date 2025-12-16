# 快速参考

## 🎯 新增功能速查

### Rastrigin 函数

**运行命令：**
```bash
cd synthetic_and_adversarial
python run.py --config config/rastrigin.yaml
```

**在配置中使用：**
```yaml
func_name: rastrigin
```

---

### 两点式（TwoPoint）优化器

**运行命令：**
```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic-twopoint.yaml
```

**在配置中使用：**
```yaml
optimizers:
  - twopoint
```

---

## 📋 配置文件列表

只有 **3 个核心配置文件**（保持原始仓库结构）：

- `config/synthetic.yaml` - 合成函数测试（支持所有优化器和函数）
- `config/synthetic-baseline.yaml` - 基线方法对比（zoo, reinforce）
- `config/adversarial.yaml` - 黑盒对抗攻击

所有新功能（rastrigin, es, twopoint）都可通过这3个配置文件使用！

---

## 🔧 优化器选项

在任何配置文件的 `optimizers` 列表中可用：

```yaml
optimizers:
  # ===== 真实梯度 =====
  - fo          # 真实梯度（仅合成函数）

  # ===== ES 家族 =====
  - es          # 纯ES（无基线）
  - vanilla     # ES + 单点基线（前向差分）
  - rl          # ES + fitness shaping（排序变换）
  - zoo         # ES + 可配置基线（需要 baseline 参数）
  - reinforce   # REINFORCE + 基线（需要 baseline 参数）

  # ===== 自适应协方差 =====
  - xnes        # xNES（完整协方差矩阵，O(d²)）✨
  - sepcmaes    # Sep-CMA-ES（对角协方差，O(d)，高维）✨

  # ===== 两点式 =====
  - twopoint    # 两点式 ZO（中心差分）

  # ===== 查询复用 =====
  - zoar        # ZoAR（带历史）
  - zoar_0      # ZoAR（无历史）
  - relizo      # ReLIZO
  - zohs        # ZOHS
  - zohs_expavg # ZOHS 指数平均
```

---

## 📊 合成函数选项

在配置文件中设置 `func_name`：

```yaml
func_name: ackley      # Ackley 函数
# 或
func_name: levy        # Levy 函数
# 或
func_name: rosenbrock  # Rosenbrock 函数
# 或
func_name: quadratic   # 二次函数
# 或
func_name: rastrigin   # Rastrigin 函数 ✨ 新增
```

---

## 🚀 快速测试组合

### 1. 合成函数测试（默认）
```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml
```
**对比：** vanilla, zoar_0, zoar, relizo, zohs

**启用其他优化器：** 编辑 `synthetic.yaml`，取消注释：
```yaml
optimizers:
  # - es        # 取消注释启用纯ES
  - vanilla
  # - twopoint  # 取消注释启用两点式
  - zoar
```

### 2. Rastrigin 函数测试
编辑 `config/synthetic.yaml`：
```yaml
func_name: rastrigin  # 改为 rastrigin
```
然后运行：
```bash
python run.py --config config/synthetic.yaml
```

### 3. ES 方法对比
编辑 `config/synthetic.yaml`：
```yaml
optimizers:
  - es       # 纯ES
  - vanilla  # ES + 基线
  - rl       # ES + 排序
  - zoar     # ZoAR
```
运行：
```bash
python run.py --config config/synthetic.yaml
```

### 4. 单点式 vs 两点式对比
编辑 `config/synthetic.yaml`：
```yaml
optimizers:
  - vanilla   # 单点式
  - twopoint  # 两点式
  - zoar
```

### 5. 基线方法测试
```bash
python run.py --config config/synthetic-baseline.yaml
```
**对比：** vanilla, twopoint, zoar, relizo

### 6. 对抗攻击测试
```bash
python run.py --config config/adversarial.yaml
```
**对比：** vanilla, zoar_0, zoar, relizo, zohs

在配置中启用 twopoint：
```yaml
optimizers:
  - vanilla
  - twopoint  # 取消注释
  - zoar
```

---

## 📖 详细文档

- `Docx/Rastrigin.md` - Rastrigin 函数数学定义
- `Docx/Rastrigin_usage.md` - Rastrigin 使用说明
- `Docx/ZO_TwoPoint.md` - 两点式参考代码
- `Docx/ZO_TwoPoint_usage.md` - 两点式详细说明
- `Docx/implementation_summary.md` - 完整实现总结
- `CLAUDE.md` - 项目整体文档

---

## 💡 关键参数说明

```yaml
num_queries: 10   # 查询数量
                  # - vanilla: 使用 10 个方向
                  # - twopoint: 使用 5 个方向（每方向2次查询）

mu: 0.05          # 扰动系数（zo_eps）

num_histories: 5  # 历史梯度数量（ZoAR, ZOHS 使用）

update_rule: radazo  # 更新规则
                     # - sgd: SGD
                     # - adam: Adam
                     # - radazo: RadAZO（推荐）
```

---

## 📁 结果存储

结果自动保存到：
- 合成函数：`results/synthetic/`
- 对抗攻击：`results/attack/`

文件名格式：
```
{func_name}_{optimizer}_{update_rule}_d{dim}_ni{iterations}_lr{lr}_nq{queries}_mu{mu}_nh{histories}_s{seed}.pt
```

使用 `torch.load()` 加载结果文件查看优化历史。
