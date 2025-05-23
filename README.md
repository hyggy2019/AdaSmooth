# Zeroth-Order Optimization is Secretly Single-Step Policy Optimization

This is official implementation for the paper "Zeroth-Order Optimization is Secretly Single-Step Policy Optimization"

## Project Overview

This project reveals a fundamental connection between Zeroth-Order Optimization (ZOO) and single-step Policy Optimization (PO) in Reinforcement Learning, showing that commonly used ZOO objectives and gradient estimators are mathematically equivalent to their PO counterparts. Building on this insight, we introduce ZoAR, a novel ZOO algorithm that incorporates query reuse and PO-inspired baseline techniques, achieving superior performance across diverse tasks.

## Dependencies
- Python 3.12
- PyTorch 2.6
- NumPy
- Matplotlib

For memory-efficient LLM fine-tuning, all necessary packages are provided in the `mezo/requirements.txt` file.

## Usage

### Synthetic Function and Black-Box Adversarial Attack

```bash
cd synthetic_and_adversarial
python run.py --config configs/synthetic.yaml # or configs/adversarial.yaml
```

### Memory-Efficient LLM Fine-Tuning

```bash
cd mezo
bash mezo.sh
```

## Citation

TODO: