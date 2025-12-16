"""
Test different divergence functionals with AdaSmoothES v2.

Compares: KL, Reverse KL, Chi-squared, Rényi, Tsallis, Huber divergences.

Author: Divergence comparison testing
"""

import torch
import argparse
from pathlib import Path
from easydict import EasyDict

from model.synthetic_functions import get_synthetic_funcs
from utils import get_optimizer


def test_divergences(
    func_name='rosenbrock',
    dimension=1000,
    num_iterations=10000,
    num_queries=10,
    seed=456,
    save_dir='results/divergence_types'
):
    """
    Test different divergence types with AdaSmoothES v2.

    Divergences tested:
    1. KL (forward KL) - default, Boltzmann weights
    2. Reverse KL - more robust to outliers
    3. Chi-squared - quadratic penalty
    4. Rényi (α=2.0) - interpolates between divergences
    5. Rényi (α=0.5) - more exploratory
    6. Tsallis (q=2.0) - heavy-tailed
    7. Huber (δ=1.0) - robust to large deviations
    """

    # Divergence configurations
    divergence_configs = [
        {
            'name': 'KL',
            'divergence': 'kl',
            'kwargs': {},
            'desc': 'Forward KL (Boltzmann)'
        },
        {
            'name': 'ReverseKL',
            'divergence': 'reverse_kl',
            'kwargs': {},
            'desc': 'Reverse KL (robust)'
        },
        {
            'name': 'ChiSquared',
            'divergence': 'chi2',
            'kwargs': {},
            'desc': 'Chi-squared (quadratic)'
        },
        {
            'name': 'Renyi_a2.0',
            'divergence': 'renyi',
            'kwargs': {'renyi_alpha': 2.0},
            'desc': 'Rényi (α=2.0, exploitative)'
        },
        {
            'name': 'Renyi_a0.5',
            'divergence': 'renyi',
            'kwargs': {'renyi_alpha': 0.5},
            'desc': 'Rényi (α=0.5, exploratory)'
        },
        {
            'name': 'Tsallis_q2.0',
            'divergence': 'tsallis',
            'kwargs': {'tsallis_q': 2.0},
            'desc': 'Tsallis (q=2.0, heavy-tail)'
        },
        {
            'name': 'Huber_d1.0',
            'divergence': 'huber',
            'kwargs': {'huber_delta': 1.0},
            'desc': 'Huber (δ=1.0, outlier-robust)'
        },
    ]

    print("=" * 80)
    print("DIVERGENCE FUNCTIONAL COMPARISON")
    print("=" * 80)
    print(f"Function: {func_name}")
    print(f"Dimension: {dimension}")
    print(f"Iterations: {num_iterations}")
    print(f"K (num_queries): {num_queries}")
    print(f"Seed: {seed}")
    print(f"Optimizer: AdaSmoothESv2")
    print("=" * 80)

    results = []

    for config in divergence_configs:
        print(f"\nTesting {config['name']}: {config['desc']}")
        print("-" * 80)

        # Set random seed
        torch.manual_seed(seed)

        # Create parameter
        param = torch.randn(dimension, requires_grad=False)

        # Get function
        func = get_synthetic_funcs(func_name, param)

        # Create args
        args = EasyDict({
            'lr': 0.001,
            'betas': [0.9, 0.99],
            'epsilon': 1e-8,
            'num_queries': num_queries,
            'mu': 0.05,
            'sigma': 0.05,
            'num_iterations': num_iterations,
            'baseline': 'mean',
            'ema_alpha': 0.1,
            # Divergence configuration
            'divergence': config['divergence'],
            # Temperature schedule (use polynomial like v1)
            'temperature_schedule': 'polynomial',
            'beta_init': 10.0,
            'beta_decay': 0.001,
            'beta_min': 0.01,
            'poly_power': 1.0,
        })

        # Add divergence-specific kwargs
        for key, value in config['kwargs'].items():
            setattr(args, key, value)

        # Create optimizer
        optimizer = get_optimizer('adasmooth_es_v2', params=func.parameters(), args=args)

        # Optimization loop
        history = []

        for t in range(num_iterations):
            def closure():
                return func()

            loss = optimizer.step(closure)
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            history.append(loss_val)

            if t % 1000 == 0 or t == num_iterations - 1:
                print(f"  Iter {t:5d}: Loss = {loss_val:12.2f}")

        final_loss = history[-1]
        print(f"\n✅ {config['name']}: Final Loss = {final_loss:.2f}")

        # Save results
        results.append({
            'name': config['name'],
            'desc': config['desc'],
            'config': config,
            'history': history,
            'final_loss': final_loss
        })

        # Save individual result file
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename with divergence info
        divergence_suffix = config['divergence']
        if config['kwargs']:
            for k, v in config['kwargs'].items():
                k_short = k.replace('renyi_', '').replace('tsallis_', '').replace('huber_', '')
                divergence_suffix += f"_{k_short}{v}"

        filename = f"{func_name}_adasmooth_es_v2_div{divergence_suffix}_d{dimension}_ni{num_iterations}_nq{num_queries}_s{seed}.pt"
        filepath = Path(save_dir) / filename
        torch.save(history, filepath)
        print(f"  Saved to: {filepath}")

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Rank':<6} {'Divergence':<20} {'Final Loss':>12} {'vs Best':>10} {'Status'}")
    print("-" * 80)

    # Sort by final loss
    results_sorted = sorted(results, key=lambda x: x['final_loss'])
    best_loss = results_sorted[0]['final_loss']

    for i, result in enumerate(results_sorted, 1):
        loss = result['final_loss']
        vs_best_pct = ((loss / best_loss - 1) * 100)
        status = "🏆" if i == 1 else ("✅" if vs_best_pct < 5 else ("⚠️" if vs_best_pct < 10 else "❌"))

        print(f"{i:<6} {result['name']:<20} {loss:12.2f} {vs_best_pct:>9.1f}% {status}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    best_result = results_sorted[0]
    worst_result = results_sorted[-1]

    print(f"✅ Best Divergence: {best_result['name']}")
    print(f"   Description: {best_result['desc']}")
    print(f"   Final Loss: {best_result['final_loss']:.2f}")

    print(f"\n❌ Worst Divergence: {worst_result['name']}")
    print(f"   Description: {worst_result['desc']}")
    print(f"   Final Loss: {worst_result['final_loss']:.2f}")

    improvement = ((worst_result['final_loss'] - best_result['final_loss']) / worst_result['final_loss']) * 100
    print(f"\n📈 Improvement (Best vs Worst): {improvement:.1f}%")

    # Top performers (within 5% of best)
    top_performers = [r for r in results_sorted if r['final_loss'] < best_result['final_loss'] * 1.05]
    print(f"\n✅ Top Performers (within 5% of best): {len(top_performers)}/{len(results_sorted)}")
    for result in top_performers:
        print(f"   - {result['name']}: {result['desc']}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {save_dir}/")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Different Divergences')
    parser.add_argument('--func', default='rosenbrock', help='Test function')
    parser.add_argument('--dim', type=int, default=1000, help='Dimension')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--queries', type=int, default=10, help='Number of queries')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--save-dir', default='results/divergence_types', help='Save directory')

    args = parser.parse_args()

    test_divergences(
        func_name=args.func,
        dimension=args.dim,
        num_iterations=args.iters,
        num_queries=args.queries,
        seed=args.seed,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
