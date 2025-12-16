# 实现总结

## 本次添加的功能

### 1. Rastrigin 函数实现
### 2. 两点式（Two-Point）ZO 梯度估计器实现
### 3. 纯ES（Evolution Strategies）优化器实现 ✨

---

## 一、Rastrigin 函数

### 修改的文件

1. **`synthetic_and_adversarial/model/synthetic_functions.py`**
   - 添加 `Rastrigin` 类（第 56-63 行）
   - 注册到 `get_synthetic_funcs()` 字典

2. **`synthetic_and_adversarial/config/rastrigin.yaml`** ✨ 新建
   - Rastrigin 函数专用配置文件

3. **`synthetic_and_adversarial/config/synthetic.yaml`**
   - 更新注释，添加 "rastrigin" 选项

4. **`CLAUDE.md`**
   - 在两处添加 Rastrigin 函数说明

5. **`Docx/Rastrigin.md`**
   - 已存在的数学定义和性质说明

6. **`Docx/Rastrigin_usage.md`** ✨ 新建
   - 详细的使用说明和示例

### 使用方法

```bash
cd synthetic_and_adversarial
python run.py --config config/rastrigin.yaml
```

---

## 二、两点式（Two-Point）ZO 梯度估计器

### 核心原理

**单点式（Vanilla）**：前向差分
```
∇f(θ) ≈ (f(θ + μu) - f(θ)) / μ
```

**两点式（TwoPoint）**：中心差分
```
∇f(θ) ≈ (f(θ + μu) - f(θ - μu)) / (2μ)
```

### 修改的文件

1. **`synthetic_and_adversarial/optimizer/zo.py`**
   - 添加 `TwoPointMatched` 类（第 429-475 行）
   - 实现中心差分梯度估计
   - 自动匹配查询预算（使用 num_queries//2 个方向）

2. **`synthetic_and_adversarial/utils.py`**
   - 导入 `TwoPointMatched`（第 16 行）
   - 在 `get_optimizer()` 中注册 "twopoint" 选项（第 70-71 行）
   - 更新可用优化器列表（第 73 行）

3. **配置文件** ✨ 新建
   - `config/synthetic-twopoint.yaml` - 通用合成函数对比
   - `config/adversarial-twopoint.yaml` - 对抗攻击场景
   - `config/rastrigin-twopoint.yaml` - Rastrigin + 两点式

4. **`CLAUDE.md`**
   - 在优化器列表中添加 `twopoint`
   - 在架构说明中详细解释两点式原理
   - 在配置说明中添加 num_queries 的双重含义

5. **`Docx/ZO_TwoPoint.md`**
   - 已存在的参考实现代码

6. **`Docx/ZO_TwoPoint_usage.md`** ✨ 新建
   - 详细的使用说明和原理对比
   - 包含查询预算匹配说明

### 使用方法

```bash
# 通用合成函数对比（单点式 vs 两点式）
cd synthetic_and_adversarial
python run.py --config config/synthetic-twopoint.yaml

# Rastrigin 函数 + 两点式
python run.py --config config/rastrigin-twopoint.yaml

# 对抗攻击 + 两点式
python run.py --config config/adversarial-twopoint.yaml
```

### 在任何配置中使用

在 YAML 配置文件的 `optimizers` 列表中添加 `twopoint`：

```yaml
optimizers:
  - vanilla    # 单点式基线
  - twopoint   # 两点式（新增）
  - zoar       # ZoAR
  - relizo     # ReLIZO
```

---

## 文件清单

### 新建文件（8个）
1. `synthetic_and_adversarial/config/rastrigin.yaml`
2. `synthetic_and_adversarial/config/synthetic-twopoint.yaml`
3. `synthetic_and_adversarial/config/adversarial-twopoint.yaml`
4. `synthetic_and_adversarial/config/rastrigin-twopoint.yaml`
5. `Docx/Rastrigin_usage.md`
6. `Docx/ZO_TwoPoint_usage.md`
7. `Docx/implementation_summary.md` (本文件)

### 修改文件（5个）
1. `synthetic_and_adversarial/model/synthetic_functions.py`
2. `synthetic_and_adversarial/optimizer/zo.py`
3. `synthetic_and_adversarial/utils.py`
4. `synthetic_and_adversarial/config/synthetic.yaml`
5. `CLAUDE.md`

---

## 技术特点

### Rastrigin 函数
- 高度多峰（大量局部最小值）
- 全局最小值：f(0) = 0
- 测试域：x_i ∈ [-5.12, 5.12]
- 公式：f(x) = 10n + Σ(x_i² - 10·cos(2πx_i))

### TwoPoint 优化器
- 继承自 `ZerothOrderOptimizer`
- 支持所有更新规则（SGD, Adam, RadAZO）
- 查询预算与 Vanilla 相同（1 + num_queries）
- 使用 num_queries//2 个方向，每方向 2 次查询
- 理论上比单点式更精确（二阶近似）

---

## 验证测试

所有实现已完成，可以通过以下方式测试：

```bash
# 测试 Rastrigin 函数
cd synthetic_and_adversarial
python run.py --config config/rastrigin.yaml

# 测试两点式方法
python run.py --config config/synthetic-twopoint.yaml

# 综合测试（Rastrigin + TwoPoint）
python run.py --config config/rastrigin-twopoint.yaml
```

结果将保存在 `results/synthetic/` 或 `results/attack/` 目录中。

---

## 三、纯ES（Evolution Strategies）优化器

### 核心原理

**纯ES（论文原始形式）**：不减基线
```
∇f(θ) ≈ (1/nσ) Σ F(θ + σεi) · εi
```

**与其他ES变体对比：**
- **Vanilla**: ES + 基线 F(θ)
- **RL**: ES + fitness shaping（排序）
- **ZOO/REINFORCE**: ES + 可配置基线

### 修改的文件

1. **`synthetic_and_adversarial/optimizer/zo.py`**
   - 添加 `ES` 类（第 429-465 行）
   - 实现无基线的梯度估计
   - 查询成本：n（无需额外基线查询）

2. **`synthetic_and_adversarial/utils.py`**
   - 导入 ES 类
   - 在 `get_optimizer()` 中注册 "es"
   - 更新可用优化器列表

3. **配置文件合并 ✨**
   - 合并 `synthetic-twopoint.yaml` → `synthetic-baseline.yaml`
   - 删除冗余的 `synthetic-twopoint.yaml`
   - 新的 `synthetic-baseline.yaml` 包含所有优化器

4. **新配置文件** ✨
   - `config/es-comparison.yaml` - ES方法对比
   - `config/rastrigin-es.yaml` - Rastrigin + ES

5. **`CLAUDE.md`**
   - 在架构说明中添加ES家族分类
   - 更新配置参数说明
   - 添加数学公式

6. **`Docx/ES_usage.md`** ✨ 新建
   - ES方法族详细说明（5种变体）
   - 数学推导和方差分析
   - 配置示例和性能建议

7. **`Docx/quick_reference.md`**
   - 在优化器列表中添加ES
   - 添加ES测试组合

8. **`Docx/ES_implementation_summary.md`** ✨ 新建
   - ES实现的完整总结
   - 方法对比表
   - 验证方法

### 使用方法

```bash
# ES 方法对比
cd synthetic_and_adversarial
python run.py --config config/es-comparison.yaml

# Rastrigin + ES
python run.py --config config/rastrigin-es.yaml

# 综合基线测试（在配置中启用ES）
python run.py --config config/synthetic-baseline.yaml
```

### 在任何配置中使用

在 YAML 配置文件的 `optimizers` 列表中添加 `es`：

```yaml
optimizers:
  - es       # 纯ES（无基线，高方差）
  - vanilla  # ES + 基线（推荐）
  - rl       # ES + 排序
```

### ES 方法对比

| 方法 | 基线 | 方差 | 查询成本 |
|------|------|------|----------|
| ES (纯) | 无 | 最高 | n |
| Vanilla | F(θ) | 高 | 1+n |
| ZOO-avg | 均值 | 中 | 1+n |
| RL | 排序 | 低 | 1+n |
| ZoAR | 历史 | 更低 | n（复用） |
| TwoPoint | 对称 | 最低 | 1+n |
