"""
Test script for AdaSmoothES v2: comparing different divergences and temperature schedules.

This script systematically tests:
1. Different divergences (KL, Reverse KL, χ², Rényi, Tsallis, Huber)
2. Different temperature schedules (constant, polynomial, exponential, cosine, cyclic)
3. Combinations of both

Results are saved to help identify best configurations.
"""

import torch
import yaml
from easydict import EasyDict
import time
from pathlib import Path

from model.synthetic_functions import get_function
from utils import get_optimizer, set_seed


def run_single_test(
    func_name: str,
    dimension: int,
    num_iterations: int,
    num_queries: int,
    divergence: str,
    temperature_schedule: str,
    seed: int = 456,
    **kwargs
):
    """
    Run a single optimization test.

    Args:
        func_name: Function name ('rosenbrock', 'ackley', etc.)
        dimension: Problem dimension
        num_iterations: Number of optimization iterations
        num_queries: Number of queries per iteration
        divergence: Divergence type
        temperature_schedule: Temperature schedule type
        seed: Random seed
        **kwargs: Additional configuration (beta_init, renyi_alpha, etc.)

    Returns:
        Dictionary with results
    """
    set_seed(seed)

    # Create function
    func = get_function(func_name, dimension)
    param = torch.randn(dimension, requires_grad=False)

    # Create args
    args = EasyDict({
        'num_queries': num_queries,
        'mu': 0.05,
        'divergence': divergence,
        'temperature_schedule': temperature_schedule,
        'baseline': 'mean',
        'num_iterations': num_iterations,
        **kwargs
    })

    # Create optimizer
    optimizer = get_optimizer('adasmooth_es_v2', params=[param], args=args)

    # Optimization loop
    start_time = time.time()
    losses = []

    def closure():
        loss = func(param)
        return loss

    for iteration in range(num_iterations):
        loss = optimizer.step(closure)
        losses.append(loss)

        if iteration % 1000 == 0:
            print(f"  Iter {iteration:5d}: Loss = {loss:.6f}, "
                  f"β = {optimizer._get_beta():.4f}, σ = {optimizer.sigma:.6f}")

    elapsed = time.time() - start_time
    final_loss = closure().item()

    result = {
        'divergence': divergence,
        'temperature_schedule': temperature_schedule,
        'final_loss': final_loss,
        'time': elapsed,
        'losses': losses,
        'config': args
    }

    return result


def test_divergences(func_name='rosenbrock', dimension=1000, num_iterations=10000, num_queries=24):
    """
    Test different divergences with fixed polynomial temperature schedule.
    """
    print("\n" + "="*80)
    print(f"Testing Divergences on {func_name} (d={dimension}, K={num_queries})")
    print("="*80)

    divergences = ['kl', 'reverse_kl', 'chi2', 'renyi', 'tsallis', 'huber']
    results = []

    for div in divergences:
        print(f"\n--- Testing {div.upper()} divergence ---")

        # Special kwargs for specific divergences
        kwargs = {}
        if div == 'renyi':
            kwargs['renyi_alpha'] = 2.0
        elif div == 'tsallis':
            kwargs['tsallis_q'] = 2.0
        elif div == 'huber':
            kwargs['huber_delta'] = 1.0

        result = run_single_test(
            func_name=func_name,
            dimension=dimension,
            num_iterations=num_iterations,
            num_queries=num_queries,
            divergence=div,
            temperature_schedule='polynomial',
            beta_init=10.0,
            beta_decay=0.001,
            **kwargs
        )

        results.append(result)
        print(f"  Final loss: {result['final_loss']:.2f}")
        print(f"  Time: {result['time']:.2f}s")

    # Sort by final loss
    results.sort(key=lambda x: x['final_loss'])

    print("\n" + "-"*80)
    print("Divergence Ranking (best to worst):")
    print("-"*80)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['divergence']:12s}: {r['final_loss']:10.2f}  ({r['time']:.2f}s)")

    return results


def test_temperature_schedules(func_name='rosenbrock', dimension=1000, num_iterations=10000, num_queries=24):
    """
    Test different temperature schedules with fixed KL divergence.
    """
    print("\n" + "="*80)
    print(f"Testing Temperature Schedules on {func_name} (d={dimension}, K={num_queries})")
    print("="*80)

    schedules = [
        ('polynomial', {'beta_init': 10.0, 'beta_decay': 0.001, 'poly_power': 1.0}),
        ('exponential', {'beta_init': 10.0, 'beta_decay': 0.001}),
        ('cosine', {'beta_init': 10.0, 'beta_min': 0.1}),
        ('linear', {'beta_init': 10.0, 'beta_min': 0.1}),
        ('constant', {'beta_init': 5.0}),
        ('cyclic', {'beta_min': 5.0, 'beta_max': 15.0, 'cycle_length': 500, 'cyclic_mode': 'triangular'}),
    ]

    results = []

    for schedule_name, kwargs in schedules:
        print(f"\n--- Testing {schedule_name.upper()} schedule ---")

        result = run_single_test(
            func_name=func_name,
            dimension=dimension,
            num_iterations=num_iterations,
            num_queries=num_queries,
            divergence='kl',
            temperature_schedule=schedule_name,
            **kwargs
        )

        results.append(result)
        print(f"  Final loss: {result['final_loss']:.2f}")
        print(f"  Time: {result['time']:.2f}s")

    # Sort by final loss
    results.sort(key=lambda x: x['final_loss'])

    print("\n" + "-"*80)
    print("Temperature Schedule Ranking (best to worst):")
    print("-"*80)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['temperature_schedule']:15s}: {r['final_loss']:10.2f}  ({r['time']:.2f}s)")

    return results


def test_best_combinations(func_name='rosenbrock', dimension=1000, num_iterations=10000, num_queries=24):
    """
    Test promising combinations of divergences and schedules.
    """
    print("\n" + "="*80)
    print(f"Testing Best Combinations on {func_name} (d={dimension}, K={num_queries})")
    print("="*80)

    # Promising combinations based on theory
    combinations = [
        ('kl', 'polynomial', {'beta_init': 10.0, 'beta_decay': 0.001}),
        ('kl', 'cosine', {'beta_init': 10.0, 'beta_min': 0.1}),
        ('reverse_kl', 'polynomial', {'beta_init': 10.0, 'beta_decay': 0.001}),
        ('chi2', 'polynomial', {'beta_init': 10.0, 'beta_decay': 0.001}),
        ('renyi', 'polynomial', {'beta_init': 10.0, 'beta_decay': 0.001, 'renyi_alpha': 2.0}),
        ('huber', 'cosine', {'beta_init': 10.0, 'beta_min': 0.1, 'huber_delta': 1.0}),
    ]

    results = []

    for div, schedule, kwargs in combinations:
        print(f"\n--- Testing {div.upper()} + {schedule.upper()} ---")

        result = run_single_test(
            func_name=func_name,
            dimension=dimension,
            num_iterations=num_iterations,
            num_queries=num_queries,
            divergence=div,
            temperature_schedule=schedule,
            **kwargs
        )

        results.append(result)
        print(f"  Final loss: {result['final_loss']:.2f}")
        print(f"  Time: {result['time']:.2f}s")

    # Sort by final loss
    results.sort(key=lambda x: x['final_loss'])

    print("\n" + "-"*80)
    print("Combination Ranking (best to worst):")
    print("-"*80)
    for i, r in enumerate(results, 1):
        combo = f"{r['divergence']} + {r['temperature_schedule']}"
        print(f"{i}. {combo:30s}: {r['final_loss']:10.2f}  ({r['time']:.2f}s)")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Test AdaSmoothES v2 configurations')
    parser.add_argument('--test', choices=['divergences', 'schedules', 'combinations', 'all'],
                       default='all', help='Which test to run')
    parser.add_argument('--func', default='rosenbrock', help='Test function')
    parser.add_argument('--dim', type=int, default=1000, help='Dimension')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--queries', type=int, default=24, help='Number of queries')

    args = parser.parse_args()

    if args.test in ['divergences', 'all']:
        test_divergences(args.func, args.dim, args.iters, args.queries)

    if args.test in ['schedules', 'all']:
        test_temperature_schedules(args.func, args.dim, args.iters, args.queries)

    if args.test in ['combinations', 'all']:
        test_best_combinations(args.func, args.dim, args.iters, args.queries)


if __name__ == '__main__':
    main()
