"""
Comprehensive comparison of ZO optimizers including AdaSmoothES with different configurations.

This script:
1. Tests AdaSmoothES with all divergences and schedulers
2. Compares with vanilla, zoar, relizo, zohs, sepcmaes
3. Generates results for plotting

Author: Comprehensive ZO comparison
"""

import torch
import time
import yaml
from pathlib import Path
from easydict import EasyDict
from model.synthetic_functions import get_synthetic_funcs
from utils import get_optimizer, set_seed

def run_single_experiment(
    func_name: str,
    optimizer_name: str,
    config: dict,
    dimension: int = 1000,
    num_iterations: int = 10000,
    seed: int = 456,
    save_dir: str = "results/synthetic"
):
    """Run a single optimization experiment."""
    set_seed(seed)

    # Create parameter and function
    param = torch.randn(dimension, requires_grad=False)
    func = get_synthetic_funcs(func_name, param)

    # Create args
    args = EasyDict({
        'num_queries': 10,
        'mu': 0.05,
        'lr': 0.001,
        'betas': [0.9, 0.99],
        'epsilon': 1e-8,
        'update_rule': 'radazo',
        'baseline': 'mean',
        'num_iterations': num_iterations,
        'num_histories': 5,
        'population_size': 10,  # For SepCMAES
        **config
    })

    # Create optimizer
    optimizer = get_optimizer(optimizer_name, params=func.parameters(), args=args)

    # Training loop
    history = []
    start_time = time.time()

    # Record initial loss
    with torch.no_grad():
        history.append(func().item())

    for iteration in range(num_iterations):
        loss = optimizer.step(lambda: func())
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        history.append(loss)

        if iteration % 2000 == 0:
            print(f"  Iter {iteration:5d}: Loss = {loss:.2f}")

    elapsed = time.time() - start_time

    # Save results
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename
    config_str = "_".join([f"{k}{v}" for k, v in sorted(config.items()) if k not in ['num_queries', 'mu', 'lr', 'betas', 'epsilon', 'update_rule', 'baseline', 'num_iterations', 'num_histories', 'population_size']])
    if config_str:
        config_str = "_" + config_str

    filename = f"{func_name}_{optimizer_name}{config_str}_d{dimension}_ni{num_iterations}_nq{args.num_queries}_mu{args.mu}_s{seed}.pt"
    torch.save(history, f"{save_dir}/{filename}")

    print(f"  Final loss: {history[-1]:.2f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Saved to: {filename}\n")

    return history, elapsed


def test_divergences_and_schedulers(
    func_name: str = 'rosenbrock',
    dimension: int = 1000,
    num_iterations: int = 10000,
    seed: int = 456
):
    """Test all combinations of divergences and beta schedulers for AdaSmoothES v2."""

    print("="*80)
    print("Testing AdaSmoothES with Different Divergences and Schedulers")
    print("="*80)

    # Note: We use AdaSmoothES v1 (not v2) because it has adaptive_beta_scheduler support
    # v2 uses TemperatureSchedule which is different

    # Best configurations we've found
    configs = [
        # Best for K=10
        {'name': 'CMA Match + Decay', 'adaptive_beta': 'cma_match', 'cma_decay': 0.001},

        # Top performers
        {'name': 'Adaptive Std c=1.0', 'adaptive_beta': 'std', 'c_beta': 1.0},
        {'name': 'Entropy Target 0.5', 'adaptive_beta': 'entropy_target', 'target_eff_ratio': 0.5},
        {'name': 'Entropy Target 0.3', 'adaptive_beta': 'entropy_target', 'target_eff_ratio': 0.3},

        # Fixed baseline
        {'name': 'Fixed Polynomial', 'beta_init': 10.0, 'beta_decay': 0.001, 'beta_schedule': 'polynomial'},
    ]

    results = []
    for config in configs:
        config_name = config.pop('name')
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print(f"{'='*60}")

        history, elapsed = run_single_experiment(
            func_name=func_name,
            optimizer_name='adasmooth_es',
            config=config,
            dimension=dimension,
            num_iterations=num_iterations,
            seed=seed,
            save_dir="results/synthetic_divergences"
        )

        results.append({
            'name': config_name,
            'final_loss': history[-1],
            'time': elapsed
        })

    # Print summary
    print("\n" + "="*80)
    print("DIVERGENCE/SCHEDULER COMPARISON SUMMARY")
    print("="*80)
    for r in sorted(results, key=lambda x: x['final_loss']):
        print(f"{r['name']:<30} Loss: {r['final_loss']:8.2f}  Time: {r['time']:.2f}s")

    return results


def run_full_comparison(
    func_name: str = 'rosenbrock',
    dimension: int = 1000,
    num_iterations: int = 10000,
    seed: int = 456
):
    """Run full comparison of all optimizers."""

    print("\n" + "="*80)
    print("Full Optimizer Comparison (for plotting)")
    print("="*80)

    # Optimizers to compare
    experiments = [
        ('vanilla', {}, 'Vanilla ES'),
        ('zoar', {}, 'ZoAR'),
        ('relizo', {}, 'ReLIZO'),
        ('zohs', {}, 'ZoHS'),
        ('sepcmaes', {}, 'SepCMAES'),
        ('adasmooth_es', {'adaptive_beta': 'cma_match', 'cma_decay': 0.001}, 'AdaSmoothES (best)'),
    ]

    results = []
    for optimizer_name, config, label in experiments:
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        try:
            history, elapsed = run_single_experiment(
                func_name=func_name,
                optimizer_name=optimizer_name,
                config=config,
                dimension=dimension,
                num_iterations=num_iterations,
                seed=seed,
                save_dir="results/synthetic_comparison"
            )

            results.append({
                'optimizer': optimizer_name,
                'label': label,
                'final_loss': history[-1],
                'time': elapsed,
                'history': history
            })
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

    # Print final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Rank':<6} {'Method':<25} {'Final Loss':>12} {'Time (s)':>10}")
    print("-"*80)

    for i, r in enumerate(sorted(results, key=lambda x: x['final_loss']), 1):
        print(f"{i:<6} {r['label']:<25} {r['final_loss']:12.2f} {r['time']:10.2f}")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive ZO Optimizer Comparison')
    parser.add_argument('--func', default='rosenbrock', help='Test function')
    parser.add_argument('--dim', type=int, default=1000, help='Dimension')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--test', choices=['divergences', 'full', 'both'], default='both',
                       help='What to test: divergences, full comparison, or both')

    args = parser.parse_args()

    if args.test in ['divergences', 'both']:
        print("\n" + "#"*80)
        print("# TASK 1: Testing AdaSmoothES Divergences and Schedulers")
        print("#"*80)
        test_divergences_and_schedulers(
            func_name=args.func,
            dimension=args.dim,
            num_iterations=args.iters,
            seed=args.seed
        )

    if args.test in ['full', 'both']:
        print("\n" + "#"*80)
        print("# TASK 2: Full Optimizer Comparison for Plotting")
        print("#"*80)
        run_full_comparison(
            func_name=args.func,
            dimension=args.dim,
            num_iterations=args.iters,
            seed=args.seed
        )

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run plotting script to visualize results")
    print("2. Check results/synthetic_comparison/ for data files")
    print("3. Verify AdaSmoothES beats SepCMAES in the plots")


if __name__ == '__main__':
    main()
