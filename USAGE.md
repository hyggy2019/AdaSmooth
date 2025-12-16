# ZoAR 使用快速指南

## 🚀 快速开始

### 1. 运行默认实验
```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml
```

### 2. 测试 Rastrigin 函数
编辑 `config/synthetic.yaml`，修改：
```yaml
func_name: rastrigin
```

### 3. 对比 ES 方法
编辑 `config/synthetic.yaml`：
```yaml
optimizers:
  - es       # 纯ES
  - vanilla  # ES + 基线
  - rl       # ES + 排序
  - zoar
```

---

## 📁 配置文件

只有 **3 个核心配置文件**：

- `config/synthetic.yaml` - 合成函数测试 ⭐
- `config/synthetic-baseline.yaml` - 基线方法对比
- `config/adversarial.yaml` - 黑盒对抗攻击

---

## 🔧 可用优化器

通过编辑配置文件的 `optimizers` 列表使用：

```yaml
optimizers:
  # - es        # 纯ES（无基线）
  - vanilla     # ES + 基线
  # - rl        # ES + 排序
  # - twopoint  # 两点式
  - zoar        # ZoAR
  - relizo      # ReLIZO
```

**完整列表：**
- `fo` - 真实梯度
- `es` - 纯ES
- `vanilla` - ES + 基线
- `rl` - ES + fitness shaping
- `xnes` - xNES（自适应完整协方差）✨
- `sepcmaes` - Sep-CMA-ES（对角协方差，高维）✨
- `twopoint` - 两点式
- `zoo` - ZOO
- `reinforce` - REINFORCE
- `zoar` / `zoar_0` - ZoAR
- `relizo` - ReLIZO
- `zohs` / `zohs_expavg` - ZOHS

---

## 📊 测试函数

修改 `func_name`：

```yaml
func_name: levy        # 默认
func_name: ackley      # Ackley
func_name: rosenbrock  # Rosenbrock
func_name: quadratic   # 二次函数
func_name: rastrigin   # Rastrigin ✨
```

---

## 📚 详细文档

- `Docx/quick_reference.md` - 快速参考 ⭐
- `Docx/config_guide.md` - 配置文件指南 ⭐
- `Docx/ES_usage.md` - ES 方法详解
- `Docx/xNES_usage.md` - xNES 详解 ✨
- `Docx/SepCMAES_usage.md` - Sep-CMA-ES 详解 ✨
- `Docx/ZO_TwoPoint_usage.md` - TwoPoint 详解
- `Docx/Rastrigin_usage.md` - Rastrigin 函数
- `CLAUDE.md` - 项目文档

---

## 📖 更多信息

查看 `Docx/CHANGELOG.md` 了解最新更新。
