"""
Test script for Adaptive Beta Schedulers.

Based on /home/zlouyang/ZoAR/Docx/AdaSpecCMA_scheduler.md

This script systematically tests different adaptive beta scheduling strategies
and compares them with:
1. Fixed polynomial schedule (baseline)
2. SepCMAES (target to beat)

Schedulers to test:
- fixed: β = β₀/(1+γt) (current default)
- std: β = c·std(f) (self-adapting)
- std_decay: β = c·std(f)/(1+γt) (adaptive + annealing)
- cma_match: β = (f_{(μ)} - f_{(1)})/log(μ) (CMA-ES equivalent)
- entropy_target: Maintain μ_eff/K = 0.5

Author: Testing adaptive beta schedulers
"""

import torch
import time
from pathlib import Path
import numpy as np

from model.synthetic_functions import get_synthetic_funcs
from utils import get_optimizer, set_seed
from easydict import EasyDict


def run_single_test(
    optimizer_name: str,
    func_name: str,
    dimension: int,
    num_iterations: int,
    num_queries: int,
    seed: int = 456,
    **kwargs
):
    """
    Run a single optimization test.

    Args:
        optimizer_name: 'adasmooth_es' or 'sepcmaes'
        func_name: Function name
        dimension: Problem dimension
        num_iterations: Number of iterations
        num_queries: Number of queries per iteration
        seed: Random seed
        **kwargs: Additional configuration for optimizer

    Returns:
        Dictionary with results
    """
    set_seed(seed)

    # Create parameter
    param = torch.randn(dimension, requires_grad=False)

    # Create function
    func = get_synthetic_funcs(func_name, param)

    # Create args
    args = EasyDict({
        'num_queries': num_queries,
        'mu': 0.05,
        'lr': 0.001,
        'betas': [0.9, 0.99],
        'epsilon': 1e-8,
        'update_rule': 'radazo',
        'baseline': 'mean',
        'num_iterations': num_iterations,
        **kwargs
    })

    # Create optimizer - CRITICAL: must pass func.parameters(), not [param]!
    optimizer = get_optimizer(optimizer_name, params=func.parameters(), args=args)

    # Optimization loop
    start_time = time.time()
    losses = []
    betas = []  # Track beta values

    def closure():
        loss = func()
        return loss

    for iteration in range(num_iterations):
        loss = optimizer.step(closure)
        losses.append(loss)

        # Track beta for AdaSmoothES
        if hasattr(optimizer, 'history') and 'beta' in optimizer.history and optimizer.history['beta']:
            betas.append(optimizer.history['beta'][-1])

        if iteration % 1000 == 0:
            beta_str = f", β = {betas[-1]:.4f}" if betas else ""
            print(f"  Iter {iteration:5d}: Loss = {loss:.6f}{beta_str}")

    elapsed = time.time() - start_time
    final_loss = closure().item()

    result = {
        'optimizer': optimizer_name,
        'final_loss': final_loss,
        'time': elapsed,
        'losses': losses,
        'betas': betas,
        'config': kwargs
    }

    return result


def test_adaptive_schedulers(
    func_name='rosenbrock',
    dimension=1000,
    num_iterations=10000,
    num_queries=24,
    seed=456
):
    """
    Test all adaptive beta schedulers + baseline + SepCMAES.

    Returns ranked results.
    """
    print("\n" + "="*80)
    print(f"Testing Adaptive Beta Schedulers vs SepCMAES")
    print(f"Function: {func_name}, Dimension: {dimension}, Queries: {num_queries}")
    print("="*80)

    configs = [
        # Baseline: Fixed polynomial (current default)
        ('Fixed Polynomial', {
            'beta_init': 10.0,
            'beta_decay': 0.001,
            'beta_schedule': 'polynomial',
            'adaptive_beta': None
        }),

        # Adaptive: std-based (pure self-adapting)
        ('Adaptive Std (c=1.0)', {
            'adaptive_beta': 'std',
            'c_beta': 1.0
        }),

        ('Adaptive Std (c=0.5)', {
            'adaptive_beta': 'std',
            'c_beta': 0.5
        }),

        ('Adaptive Std (c=2.0)', {
            'adaptive_beta': 'std',
            'c_beta': 2.0
        }),

        # Adaptive + Decay (adaptive + forced convergence)
        ('Adaptive Std + Decay', {
            'adaptive_beta': 'std_decay',
            'c_beta': 1.0,
            'beta_decay': 0.001
        }),

        # CMA-match (match CMA-ES ranking weights)
        ('CMA Match (no decay)', {
            'adaptive_beta': 'cma_match',
            'cma_decay': 0.0
        }),

        ('CMA Match + Decay', {
            'adaptive_beta': 'cma_match',
            'cma_decay': 0.001
        }),

        # Entropy target (maintain effective sample ratio)
        ('Entropy Target (0.5)', {
            'adaptive_beta': 'entropy_target',
            'target_eff_ratio': 0.5
        }),

        ('Entropy Target (0.3)', {
            'adaptive_beta': 'entropy_target',
            'target_eff_ratio': 0.3
        }),

        # Range-based
        ('Range Based', {
            'adaptive_beta': 'range',
            'c_beta': 0.5
        }),
    ]

    results = []

    # Test AdaSmoothES with different schedulers
    for config_name, config_kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print(f"{'='*60}")

        try:
            result = run_single_test(
                optimizer_name='adasmooth_es',
                func_name=func_name,
                dimension=dimension,
                num_iterations=num_iterations,
                num_queries=num_queries,
                seed=seed,
                **config_kwargs
            )
            result['config_name'] = config_name
            results.append(result)
            print(f"  Final loss: {result['final_loss']:.2f}")
            print(f"  Time: {result['time']:.2f}s")
            if result['betas']:
                print(f"  Final β: {result['betas'][-1]:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Test SepCMAES for comparison
    print(f"\n{'='*60}")
    print(f"Testing: SepCMAES (baseline)")
    print(f"{'='*60}")

    result_sepcma = run_single_test(
        optimizer_name='sepcmaes',
        func_name=func_name,
        dimension=dimension,
        num_iterations=num_iterations,
        num_queries=num_queries,
        seed=seed,
        population_size=num_queries  # Fair comparison
    )
    result_sepcma['config_name'] = 'SepCMAES'
    results.append(result_sepcma)
    print(f"  Final loss: {result_sepcma['final_loss']:.2f}")
    print(f"  Time: {result_sepcma['time']:.2f}s")

    # Sort by final loss
    results.sort(key=lambda x: x['final_loss'])

    # Print ranking
    print("\n" + "="*80)
    print("RESULTS RANKING (best to worst)")
    print("="*80)
    print(f"{'Rank':<6} {'Method':<30} {'Final Loss':>12} {'Time (s)':>10} {'vs SepCMA':>12}")
    print("-"*80)

    sepcma_loss = result_sepcma['final_loss']

    for i, r in enumerate(results, 1):
        vs_sepcma = f"{(r['final_loss'] / sepcma_loss - 1) * 100:+.1f}%"
        marker = "🏆" if r['config_name'] == 'SepCMAES' else ("✅" if r['final_loss'] <= sepcma_loss * 1.05 else "")
        print(f"{i:<6} {r['config_name']:<30} {r['final_loss']:12.2f} {r['time']:10.2f} {vs_sepcma:>12} {marker}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    best_adasmooth = [r for r in results if 'SepCMAES' not in r['config_name']][0]
    improvement = (sepcma_loss - best_adasmooth['final_loss']) / sepcma_loss * 100

    if best_adasmooth['final_loss'] < sepcma_loss:
        print(f"✅ BEST AdaSmoothES beats SepCMAES by {improvement:.1f}%!")
        print(f"   Method: {best_adasmooth['config_name']}")
    elif best_adasmooth['final_loss'] < sepcma_loss * 1.05:
        print(f"✅ BEST AdaSmoothES matches SepCMAES (within 5%)")
        print(f"   Method: {best_adasmooth['config_name']}")
        print(f"   Gap: {-improvement:.1f}%")
    else:
        print(f"❌ Best AdaSmoothES is {-improvement:.1f}% worse than SepCMAES")
        print(f"   Method: {best_adasmooth['config_name']}")

    print(f"\nSepCMAES: {sepcma_loss:.2f}")
    print(f"Best AdaSmoothES: {best_adasmooth['final_loss']:.2f} ({best_adasmooth['config_name']})")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Test Adaptive Beta Schedulers')
    parser.add_argument('--func', default='rosenbrock', help='Test function')
    parser.add_argument('--dim', type=int, default=1000, help='Dimension')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--queries', type=int, default=24, help='Number of queries')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')

    args = parser.parse_args()

    results = test_adaptive_schedulers(
        func_name=args.func,
        dimension=args.dim,
        num_iterations=args.iters,
        num_queries=args.queries,
        seed=args.seed
    )

    # Save results
    output_file = f"/tmp/adaptive_scheduler_results_{args.func}_d{args.dim}_K{args.queries}.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Adaptive Beta Scheduler Test Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Function: {args.func}\n")
        f.write(f"Dimension: {args.dim}\n")
        f.write(f"Iterations: {args.iters}\n")
        f.write(f"Queries: {args.queries}\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"{i}. {r['config_name']}: {r['final_loss']:.2f} ({r['time']:.2f}s)\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
