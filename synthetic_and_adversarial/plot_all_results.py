"""
Plot all experimental results as specified in TODO.md

Generates:
1. Convergence curves for synthetic functions
2. Final convergence value tables
3. Adversarial attack success rates and speedup tables
4. All plots follow publication standards (Type 42 fonts)

Author: Result visualization automation
"""

import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple

# =============================================================================
# Publication-ready matplotlib configuration (as per TODO.md)
# =============================================================================

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['mathtext.fontset'] = 'cm'


# =============================================================================
# Algorithm configurations
# =============================================================================

ALGORITHMS = {
    'vanilla': {'label': 'Vanilla', 'marker': r'$\heartsuit$', 'color': '#1E90FF'},
    'zoar': {'label': 'ZoAR', 'marker': r'$\triangle$', 'color': '#FF6347'},
    'relizo': {'label': 'ReLIZO', 'marker': 'p', 'color': '#4682B4'},
    'twopoint': {'label': 'TwoPoint', 'marker': 'd', 'color': '#32CD32'},
    'zohs': {'label': 'ZoHS', 'marker': r'$\boxdot$', 'color': '#3CB371'},
    'sepcmaes': {'label': 'SepCMAES', 'marker': r'$\circ$', 'color': '#D2691E'},
    'adasmooth_es': {'label': 'AdaSmoothES', 'marker': r'$\star$', 'color': '#9400D3'},
}


# =============================================================================
# 1. Synthetic Function Plots
# =============================================================================

def plot_synthetic_convergence(
    func_name: str,
    dimension: int,
    num_iterations: int,
    num_queries: int = 10,
    mu: float = 0.05,
    seed: int = 456,
    results_dir: str = 'results/synthetic',
    save_dir: str = 'figures'
):
    """
    Plot convergence curves for a synthetic function.

    Args:
        func_name: Function name (rosenbrock, ackley, rastrigin)
        dimension: Problem dimension
        num_iterations: Number of iterations
        num_queries: Number of queries
        mu: Perturbation parameter
        seed: Random seed
        results_dir: Directory containing results
        save_dir: Directory to save figures
    """
    plt.figure(figsize=(8, 6))

    n = 16  # Number of points to plot
    histories = {}

    # Load all algorithm results
    for alg_name, config in ALGORITHMS.items():
        try:
            # Construct filename
            filename = f"{func_name}_{alg_name}_radazo_d{dimension}_ni{num_iterations}_lr0.001_nq{num_queries}_mu{mu}_nh5_s{seed}.pt"
            filepath = Path(results_dir) / filename

            if not filepath.exists():
                print(f"Warning: {filename} not found")
                continue

            history = torch.load(filepath, weights_only=True)
            histories[alg_name] = history

            # Plot
            start = 0
            end = len(history)
            interval = max((end - start) // n, 1)
            xs = torch.arange(start, end, interval)
            ys = torch.log(torch.tensor(history))[start:end:interval]

            plt.plot(xs, ys, marker=config['marker'], color=config['color'],
                    label=config['label'], linestyle='--', markersize=7.0, linewidth=2)

        except Exception as e:
            print(f"Error loading {alg_name}: {e}")
            continue

    # Formatting
    plt.xlabel('Iterations ($T$)', fontsize=14)
    plt.ylabel('Optimality Gap (log scale)', fontsize=14)
    plt.title(f'{func_name.capitalize()} (d={dimension})', fontsize=16)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f'{func_name}_d{dimension}_convergence.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

    return histories


def create_synthetic_table(
    functions: List[str],
    dimensions: List[int],
    num_queries: int = 10,
    mu: float = 0.05,
    seed: int = 456,
    results_dir: str = 'results/synthetic',
    save_dir: str = 'figures'
):
    """
    Create table of final convergence values.

    Returns:
        DataFrame with final losses for each algorithm and configuration
    """
    data = []

    for func in functions:
        for dim in dimensions:
            row = {'Function': func.capitalize(), 'Dimension': dim}

            for alg_name in ALGORITHMS.keys():
                try:
                    # Determine num_iterations based on dimension
                    if dim == 1000:
                        num_iterations = 10000
                    elif dim == 5000:
                        num_iterations = 15000
                    elif dim == 10000:
                        num_iterations = 20000
                    else:
                        num_iterations = 10000

                    filename = f"{func}_{alg_name}_radazo_d{dim}_ni{num_iterations}_lr0.001_nq{num_queries}_mu{mu}_nh5_s{seed}.pt"
                    filepath = Path(results_dir) / filename

                    if filepath.exists():
                        history = torch.load(filepath, weights_only=True)
                        final_loss = history[-1]
                        row[ALGORITHMS[alg_name]['label']] = f"{final_loss:.2f}"
                    else:
                        row[ALGORITHMS[alg_name]['label']] = "N/A"

                except Exception as e:
                    print(f"Error loading {alg_name} for {func} d={dim}: {e}")
                    row[ALGORITHMS[alg_name]['label']] = "Error"

            data.append(row)

    df = pd.DataFrame(data)

    # Save as CSV and LaTeX
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(save_dir) / 'synthetic_final_losses.csv'
    latex_path = Path(save_dir) / 'synthetic_final_losses.tex'

    df.to_csv(csv_path, index=False)
    df.to_latex(latex_path, index=False, escape=False)

    print(f"\nSaved table:")
    print(f"  CSV: {csv_path}")
    print(f"  LaTeX: {latex_path}")
    print("\nTable preview:")
    print(df.to_string(index=False))

    return df


# =============================================================================
# 2. Adversarial Attack Plots
# =============================================================================

def calculate_attack_metrics(
    dataset: str,
    threshold: float = 0.5,
    results_dir: str = 'results/attack',
    num_queries: int = 10,
    mu: float = 0.05,
    seed: int = 456
):
    """
    Calculate adversarial attack metrics.

    Returns:
        Dict with metrics for each algorithm:
        - num_iters: Number of iterations to success
        - speedup: Speedup compared to vanilla
    """
    metrics = {}

    for alg_name in ALGORITHMS.keys():
        try:
            # Find result file
            pattern = f"{dataset}_{alg_name}_*_nq{num_queries}_mu{mu}_s{seed}.pt"
            files = list(Path(results_dir).glob(pattern))

            if not files:
                print(f"Warning: No results for {alg_name} on {dataset}")
                metrics[alg_name] = {'num_iters': None, 'success': False}
                continue

            filepath = files[0]
            history = torch.load(filepath, weights_only=True)

            # Find first iteration where loss < threshold (success)
            success_iter = None
            for i, loss in enumerate(history):
                if loss < threshold:
                    success_iter = i
                    break

            if success_iter is not None:
                metrics[alg_name] = {
                    'num_iters': success_iter,
                    'final_loss': history[-1],
                    'success': True
                }
            else:
                metrics[alg_name] = {
                    'num_iters': len(history),
                    'final_loss': history[-1],
                    'success': False
                }

        except Exception as e:
            print(f"Error loading {alg_name} for {dataset}: {e}")
            metrics[alg_name] = {'num_iters': None, 'success': False}

    # Calculate speedup (relative to vanilla)
    if 'vanilla' in metrics and metrics['vanilla']['num_iters'] is not None:
        baseline = metrics['vanilla']['num_iters']
        for alg_name in metrics:
            if metrics[alg_name]['num_iters'] is not None:
                metrics[alg_name]['speedup'] = baseline / metrics[alg_name]['num_iters']
            else:
                metrics[alg_name]['speedup'] = None

    return metrics


def create_adversarial_table(
    datasets: List[str],
    save_dir: str = 'figures'
):
    """
    Create table of adversarial attack metrics (as per TODO.md format).

    Returns:
        DataFrame with metrics for each algorithm
    """
    # Calculate metrics for each dataset
    all_metrics = {}
    for dataset in datasets:
        all_metrics[dataset] = calculate_attack_metrics(dataset)

    # Create table (format from TODO.md)
    data = []

    for metric_name in ['# Iters', 'Speedup']:
        row = {'Metric': metric_name}

        for alg_name in ALGORITHMS.keys():
            if metric_name == '# Iters':
                # Average over datasets, in units of 10^2
                values = []
                for dataset in datasets:
                    if all_metrics[dataset][alg_name]['num_iters'] is not None:
                        values.append(all_metrics[dataset][alg_name]['num_iters'] / 100)

                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    row[ALGORITHMS[alg_name]['label']] = f"{mean:.2f} ± {std:.2f}"
                else:
                    row[ALGORITHMS[alg_name]['label']] = "N/A"

            elif metric_name == 'Speedup':
                # Average speedup
                values = []
                for dataset in datasets:
                    if all_metrics[dataset][alg_name]['speedup'] is not None:
                        values.append(all_metrics[dataset][alg_name]['speedup'])

                if values:
                    mean = np.mean(values)
                    row[ALGORITHMS[alg_name]['label']] = f"{mean:.2f}×"
                else:
                    row[ALGORITHMS[alg_name]['label']] = "N/A"

        data.append(row)

    df = pd.DataFrame(data)

    # Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(save_dir) / 'adversarial_metrics.csv'
    latex_path = Path(save_dir) / 'adversarial_metrics.tex'

    df.to_csv(csv_path, index=False)
    df.to_latex(latex_path, index=False, escape=False)

    print(f"\nSaved adversarial table:")
    print(f"  CSV: {csv_path}")
    print(f"  LaTeX: {latex_path}")
    print("\nTable preview:")
    print(df.to_string(index=False))

    return df


def plot_adversarial_convergence(
    dataset: str,
    results_dir: str = 'results/attack',
    save_dir: str = 'figures',
    num_queries: int = 10,
    mu: float = 0.05,
    seed: int = 456
):
    """Plot convergence curves for adversarial attack."""
    plt.figure(figsize=(8, 6))

    n = 16
    histories = {}

    for alg_name, config in ALGORITHMS.items():
        try:
            pattern = f"{dataset}_{alg_name}_*_nq{num_queries}_mu{mu}_s{seed}.pt"
            files = list(Path(results_dir).glob(pattern))

            if not files:
                print(f"Warning: No results for {alg_name} on {dataset}")
                continue

            filepath = files[0]
            history = torch.load(filepath, weights_only=True)
            histories[alg_name] = history

            # Plot
            start = 0
            end = len(history)
            interval = max((end - start) // n, 1)
            xs = torch.arange(start, end, interval)
            ys = torch.log(torch.tensor(history))[start:end:interval]

            plt.plot(xs, ys, marker=config['marker'], color=config['color'],
                    label=config['label'], linestyle='--', markersize=7.0, linewidth=2)

        except Exception as e:
            print(f"Error loading {alg_name}: {e}")
            continue

    # Formatting
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Attack Loss (log scale)', fontsize=14)
    plt.title(f'{dataset.upper()} Adversarial Attack', fontsize=16)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f'{dataset}_adversarial_convergence.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

    return histories


# =============================================================================
# Main execution
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Plot all experimental results')
    parser.add_argument('--synthetic-only', action='store_true', help='Plot only synthetic results')
    parser.add_argument('--adversarial-only', action='store_true', help='Plot only adversarial results')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--save-dir', default='figures', help='Save directory')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PLOTTING ALL RESULTS (TODO.md)")
    print("="*80)

    # Synthetic functions
    if not args.adversarial_only:
        print("\n" + "="*80)
        print("1. SYNTHETIC FUNCTIONS")
        print("="*80)

        functions = ['rosenbrock', 'ackley', 'rastrigin']
        dimensions = [1000, 5000, 10000]

        # Plot convergence curves
        for func in functions:
            for dim in dimensions:
                if dim == 1000:
                    num_iters = 10000
                elif dim == 5000:
                    num_iters = 15000
                else:
                    num_iters = 20000

                print(f"\nPlotting {func} d={dim}...")
                plot_synthetic_convergence(
                    func_name=func,
                    dimension=dim,
                    num_iterations=num_iters,
                    results_dir=f"{args.results_dir}/synthetic",
                    save_dir=args.save_dir
                )

        # Create table
        print("\nCreating synthetic results table...")
        create_synthetic_table(
            functions=functions,
            dimensions=dimensions,
            results_dir=f"{args.results_dir}/synthetic",
            save_dir=args.save_dir
        )

    # Adversarial attacks
    if not args.synthetic_only:
        print("\n" + "="*80)
        print("2. ADVERSARIAL ATTACKS")
        print("="*80)

        datasets = ['mnist', 'cifar10']

        # Plot convergence curves
        for dataset in datasets:
            print(f"\nPlotting {dataset} adversarial attack...")
            plot_adversarial_convergence(
                dataset=dataset,
                results_dir=f"{args.results_dir}/attack",
                save_dir=args.save_dir
            )

        # Create table
        print("\nCreating adversarial metrics table...")
        create_adversarial_table(
            datasets=datasets,
            save_dir=args.save_dir
        )

    print("\n" + "="*80)
    print("✅ All plots and tables generated!")
    print(f"Saved to: {args.save_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
