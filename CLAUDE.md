# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZoAR (Zeroth-Order optimization with Adaptive Reuse) is a research implementation that demonstrates the mathematical equivalence between Zeroth-Order Optimization (ZOO) and single-step Policy Optimization (PO) in Reinforcement Learning. The project implements novel ZOO algorithms with query reuse and PO-inspired baseline techniques across three domains:

1. **Synthetic function optimization** - Testing on mathematical functions (Ackley, Levy, Rosenbrock, Quadratic, Rastrigin)
2. **Black-box adversarial attacks** - Image perturbation on MNIST/CIFAR10
3. **Memory-efficient LLM fine-tuning** - Language model optimization using zeroth-order methods

## Repository Structure

```
ZoAR/
├── mezo/                           # Memory-efficient zeroth-order LLM fine-tuning
│   ├── run.py                      # Main entry point for LLM experiments
│   ├── trainer.py                  # Custom trainer extending HuggingFace Trainer
│   ├── ht_opt.py                   # Zeroth-order optimization implementation
│   ├── tasks.py                    # Task definitions (SST2, RTE, CB, BoolQ, WSC, etc.)
│   ├── templates.py                # Prompt templates for different tasks
│   ├── metrics.py                  # Evaluation metrics
│   ├── lora.py                     # LoRA parameter-efficient fine-tuning
│   ├── prefix.py                   # Prefix tuning implementation
│   ├── utils.py                    # Helper functions
│   ├── mezo.sh                     # Main script for ZOO experiments
│   ├── finetune.sh                 # Standard fine-tuning baseline
│   └── requirements.txt            # Python dependencies
│
├── synthetic_and_adversarial/      # Synthetic & adversarial attack experiments
│   ├── run.py                      # Entry point (routes based on config.exp field)
│   ├── utils.py                    # Common utilities and optimizer factory
│   ├── config/                     # YAML configuration files
│   │   ├── synthetic.yaml          # Config for synthetic function experiments
│   │   ├── synthetic-baseline.yaml # Baseline configuration
│   │   └── adversarial.yaml        # Config for adversarial attack experiments
│   ├── optimizer/                  # Optimizer implementations
│   │   ├── zo.py                   # Base ZO optimizer (vanilla, ZoAR, ZOHS, etc.)
│   │   ├── relizo_adam.py          # ReLIZO optimizer
│   │   └── fo.py                   # First-order baseline
│   ├── model/                      # Models and objective functions
│   │   ├── synthetic_functions.py  # Test functions (Ackley, Levy, Rastrigin, etc.)
│   │   ├── attack.py               # Adversarial attack objective
│   │   └── cnn.py                  # CNN for adversarial experiments
│   ├── script/                     # Experiment runners
│   │   ├── run_synthetic.py        # Synthetic function optimization loop
│   │   └── run_adversarial.py      # Adversarial attack optimization loop
│   └── data/                       # Dataset storage (MNIST, CIFAR10)
│
└── Docx/                           # Additional documentation
    ├── quick_reference.md          # Quick start guide
    ├── config_guide.md             # Configuration file guide
    └── [optimizer-specific guides] # Detailed optimizer documentation
```

## Common Commands

### Synthetic Function Optimization

```bash
cd synthetic_and_adversarial
python run.py --config config/synthetic.yaml
```

Config options in `config/synthetic.yaml`:
- `func_name`: Choose from "ackley", "levy", "rosenbrock", "quadratic", "rastrigin"
- `optimizers`: List of optimizers to compare
  - First-order: `fo` (true gradient, only for synthetic functions)
  - ES family: `es` (pure ES), `vanilla` (ES+baseline), `rl` (ES+fitness shaping)
  - Adaptive methods: `xnes` (full covariance), `sepcmaes` (diagonal covariance, high-dimensional), `adasmooth` (adaptive low-rank sampling)
  - Gradient estimators: `twopoint` (central difference)
  - Query reuse: `zoar`, `zoar_0`, `relizo`, `zohs`, `zohs_expavg`
  - Baseline variants: `zoo`, `reinforce` (require `baseline` parameter)
- `dimension`: Problem dimensionality
- `num_iterations`: Optimization steps
- `update_rule`: "sgd", "adam", or "radazo"
- `num_queries`: Number of query samples for ZO gradient estimation
  - For `es`/`vanilla`/`rl`: uses `num_queries` directions
  - For `twopoint`: uses `num_queries//2` directions (matched query budget)
- `mu`: Perturbation parameter (zo_eps)
- `num_histories`: Number of historical gradients for ZoAR, ZOHS
- `baseline`: Baseline type for ZOO/REINFORCE ("single" or "average")
- Optional parameters for specific optimizers (xNES, SepCMAES, AdaSmoothZO) - see config file comments

Results saved to `results/synthetic/`

### Black-Box Adversarial Attack

```bash
cd synthetic_and_adversarial
python run.py --config config/adversarial.yaml
```

Config options in `config/adversarial.yaml`:
- `dataset`: "mnist" or "cifar10"
- `model`: Attack target model
- `idx`: Image index to attack
- `device`: "cuda", "cpu", or "mps"
- Same optimizer options as synthetic experiments

Results saved to `results/attack/`

### Memory-Efficient LLM Fine-Tuning

```bash
cd mezo
bash mezo.sh
```

Key environment variables for `mezo.sh`:
- `OPT`: Optimization method ("vanilla" or "zoar")
- `TASK`: Task name (SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP)
- `MODEL`: HuggingFace model path (default: facebook/opt-1.3b)
- `MODE`: "lora" or "prefix" (parameter-efficient methods)
- `OPT_TYPE`: Update rule ("sgd", "adam", "radazo")
- `NUM_HISTORIES`: Number of historical gradients (default: 15)
- `BS`: Batch size
- `LR`: Learning rate
- `EPS`: ZO perturbation epsilon (zo_eps)
- `STEPS`: Max training steps
- `SEED`: Random seed

Standard fine-tuning baseline:
```bash
cd mezo
bash finetune.sh
```

Results logged to WandB and saved to `result/` directory.

## Architecture Notes

### Zeroth-Order Optimization Framework

The core ZO optimization is implemented differently for each domain:

**LLM Fine-tuning** (`mezo/ht_opt.py` and `mezo/trainer.py`):
- Custom `OurTrainer` extends HuggingFace `Trainer` class
- Implements ZO gradient estimation via forward differences
- Supports query reuse and historical gradient baselines
- Integrates with LoRA and prefix tuning for parameter efficiency

**Synthetic & Adversarial** (`synthetic_and_adversarial/optimizer/zo.py`):
- Base `ZerothOrderOptimizer` class extends `torch.optim.Optimizer`
- Subclasses implement different gradient estimators:
  - **ES family** (Evolution Strategies):
    - `ES`: Pure ES without baseline: `∇f ≈ (1/nσ) Σ F(θ+σε)·ε` (highest variance, theoretical form)
    - `Vanilla`: ES + baseline: `∇f ≈ (1/nμ) Σ [F(θ+μu) - F(θ)]·u`
    - `Reinforcement_Learning`: ES + fitness shaping (rank transformation)
    - `ZOO`/`REINFORCE`: ES with configurable baseline ("single" or "average")
  - **Adaptive covariance methods**:
    - `xNES`: Exponential Natural Evolution Strategies with full covariance matrix
      - Uses natural gradients to update search distribution
      - Adapts full covariance matrix to capture parameter correlations
      - Complexity: O(d²) per iteration, suitable for d < 5000
    - `SepCMAES`: Separable CMA-ES with diagonal covariance matrix
      - Uses `cmaes` library's SepCMA implementation
      - Diagonal covariance (learns per-dimension step sizes)
      - Complexity: O(d) per iteration, suitable for d > 5000
      - No gradient computation (ask-tell interface)
  - **Two-point estimator**:
    - `TwoPointMatched`: Central difference: `∇f ≈ (1/m) Σ [F(θ+μu) - F(θ-μu)]/(2μ)·u`
      - Uses `num_queries//2` directions to match one-point's query budget
      - Lower variance due to symmetric sampling
  - **Query reuse methods**:
    - `ZoAR`: Query reuse with historical baselines
    - `ZoAR_0`: ZoAR without history (baseline only)
    - `ZOHS`: History smoothing with exponential averaging
    - `ZOHS_expavg`: ZOHS with exponential average variant
  - **Adaptive sampling**:
    - `AdaSmoothZO`: Adaptive low-rank sampling with temperature scheduling
      - Supports polynomial, exponential, and constant schedules
      - Uses Gumbel-Softmax for gradient estimation
- Supports SGD, Adam, and RadAZO update rules
- All optimizers extend `torch.optim.Optimizer` interface

### Optimizer Selection

The `get_optimizer()` function in `utils.py` (both directories) acts as a factory:
- Maps optimizer names ("vanilla", "zoar", "relizo", etc.) to implementations
- Initializes with shared hyperparameters (lr, betas, epsilon, etc.)
- For LLM tasks, optimizer selection happens in `run.py` via `--opt_name` and `--opt_type`

### Task Handling (LLM)

Tasks are defined in `mezo/tasks.py`:
- Each task has a `Dataset` subclass (e.g., `SST2Dataset`, `RTEDataset`)
- Tasks define templates in `mezo/templates.py` for prompt formatting
- Special handling for classification vs generation tasks (Copa, ReCoRD, SQuAD, DROP set `--train_as_classification False`)
- Small datasets (CB, Copa) use reduced dev sets (100 samples)

### Configuration System

**Synthetic/Adversarial**: YAML-based configs loaded with `easydict.EasyDict`
**LLM**: Shell script environment variables + HuggingFace `TrainingArguments` dataclass

## Dependencies

- Python 3.12
- PyTorch 2.6 (note: `mezo/requirements.txt` specifies torch 2.4.0)
- For LLM experiments: Install from `mezo/requirements.txt`
  - HuggingFace transformers, datasets, accelerate
  - WandB for logging
- For synthetic/adversarial experiments:
  - PyTorch, NumPy, Matplotlib, PyYAML
  - `easydict` for config parsing
  - `cmaes` library for SepCMAES optimizer
  - No formal requirements.txt; install as needed

## Development Notes

### Running Single Task Experiments

For LLM tasks, override environment variables:
```bash
TASK=BoolQ BS=32 LR=1e-4 STEPS=3000 bash mezo.sh
```

For synthetic tasks, modify the YAML config file directly:
```bash
cd synthetic_and_adversarial
# Edit config/synthetic.yaml to change func_name, optimizers, etc.
python run.py --config config/synthetic.yaml
```

Quick optimizer comparisons - edit the `optimizers` list in the config:
```yaml
optimizers:
  - vanilla   # ES + baseline
  - twopoint  # Two-point estimator
  - zoar      # ZoAR with history
  - relizo    # ReLIZO
```

### Adding New Optimizers

1. For synthetic/adversarial: Extend `ZerothOrderOptimizer` in `synthetic_and_adversarial/optimizer/`
2. Register in `get_optimizer()` in `utils.py`
3. Add to `optimizers` list in YAML config

For LLM tasks: Modify `mezo/ht_opt.py` and add logic to `mezo/trainer.py`

### Task-Specific Considerations (LLM)

Some tasks require special handling:
- **MultiRC, ReCoRD**: Limited batch size on 80GB GPUs, use gradient accumulation
- **DROP**: Batch size of 1 required
- **Copa, ReCoRD, SQuAD, DROP**: Generation tasks, not classification
- **CB, Copa**: Small training sets (<1000 examples)

Check `mezo.sh` and `finetune.sh` for task-specific `TASK_ARGS`.

### Results and Logging

- Synthetic/adversarial: Results saved as PyTorch tensors (`.pt`) with detailed filenames encoding hyperparameters
  - File naming format: `{func_name}_{optimizer}_{update_rule}_d{dim}_ni{iterations}_lr{lr}_nq{queries}_mu{mu}_nh{histories}_s{seed}.pt`
  - Load with `torch.load()` to access optimization history
- LLM: WandB logging enabled by default, results in `result/` directory
- Use `--tag` and `--run_name` for organizing experiments

### Additional Documentation

The `Docx/` directory contains detailed documentation on specific topics:
- `quick_reference.md` - Quick start guide with common usage patterns
- `config_guide.md` - Comprehensive configuration file guide
- `ES_usage.md`, `xNES_usage.md`, `SepCMAES_usage.md`, `ZO_TwoPoint_usage.md` - Optimizer-specific guides
- `Rastrigin.md` - Rastrigin test function details
- `FINAL_SUMMARY.md`, `implementation_summary.md` - Implementation summaries
