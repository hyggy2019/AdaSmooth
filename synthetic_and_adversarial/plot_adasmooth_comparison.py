"""
Plot comprehensive comparison of ZO optimizers including AdaSmoothES.

Based on the plotting style from /home/zlouyang/ZoAR/figures.ipynb

Author: AdaSmoothES comparison plotting
"""

import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
from pathlib import Path

# Set matplotlib style (matching figures.ipynb)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["figure.figsize"] = (3.5, 3)


def plot_comparison(
    func_name: str = 'rosenbrock',
    dimension: int = 1000,
    num_iterations: int = 10000,
    num_queries: int = 10,
    mu: float = 0.05,
    seed: int = 456,
    save_dir: str = 'figures',
    show_speedup: bool = True
):
    """
    Plot optimization curves comparing different optimizers.

    Args:
        func_name: Test function name
        dimension: Problem dimension
        num_iterations: Number of iterations
        num_queries: Number of queries
        mu: Perturbation parameter
        seed: Random seed
        save_dir: Directory to save figures
        show_speedup: Whether to annotate speedup arrow
    """

    # Optimizer configurations
    optimizers = [
        {'name': 'vanilla', 'label': 'Vanilla ES', 'config_suffix': ''},
        {'name': 'zohs', 'label': 'ZoHS', 'config_suffix': ''},
        {'name': 'relizo', 'label': 'ReLIZO', 'config_suffix': ''},
        {'name': 'zoar', 'label': 'ZoAR', 'config_suffix': ''},
        {'name': 'sepcmaes', 'label': 'SepCMAES', 'config_suffix': ''},
        {'name': 'adasmooth_es', 'label': 'AdaSmoothES (CMA Match)', 'config_suffix': '_adaptive_betacma_match_cma_decay0.001'},
    ]

    # Plotting configuration (matching figures.ipynb style)
    markers = [r'$\heartsuit$', r'$\boxdot$', 'p', r'$\triangle$', r'$\circ$', r'$\star$']
    colors = ['#1E90FF', '#3CB371', '#4682B4', '#FF6347', '#D2691E', '#9400D3']
    n = 16  # Number of points to plot

    plt.figure(figsize=(6, 5))

    histories = []
    for i, opt in enumerate(optimizers):
        try:
            # Load result file
            filename = f"{func_name}_{opt['name']}{opt['config_suffix']}_d{dimension}_ni{num_iterations}_nq{num_queries}_mu{mu}_s{seed}.pt"
            filepath = Path('results/synthetic_comparison') / filename

            if not filepath.exists():
                print(f"Warning: File not found: {filepath}")
                continue

            history = torch.load(filepath, weights_only=True)
            histories.append((opt, history))

            # Prepare data for plotting
            start = 0
            end = len(history)
            interval = max((end - start) // n, 1)
            xs = torch.arange(start, end, interval)
            ys = torch.log(torch.tensor(history))[start:end:interval]

            # Plot
            plt.plot(xs, ys, marker=markers[i], color=colors[i],
                    label=opt['label'], linestyle='--', markersize=6.0)

            # Track for speedup calculation
            if opt['name'] == 'sepcmaes':
                sepcma_x_end = xs[-1]
                sepcma_y_end = ys[-1]
            elif opt['name'] == 'adasmooth_es':
                adasmooth_ys = ys
                adasmooth_xs = xs

        except Exception as e:
            print(f"Error loading {opt['name']}: {e}")
            continue

    # Compute and annotate speedup (if both SepCMAES and AdaSmoothES exist)
    if show_speedup and 'adasmooth_ys' in locals() and 'sepcma_y_end' in locals():
        # Find where AdaSmoothES reaches SepCMAES's final loss
        dist = torch.abs(adasmooth_ys - sepcma_y_end)
        ind = torch.argmin(dist)
        ada_x_at_sepcma_loss = adasmooth_xs[ind]

        speedup = sepcma_x_end / ada_x_at_sepcma_loss

        # Draw speedup arrow
        plt.annotate(
            '',  # No text
            xy=(sepcma_x_end, sepcma_y_end),  # End point
            xytext=(ada_x_at_sepcma_loss, sepcma_y_end),  # Start point
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                lw=1.3,
                shrinkA=0,
                shrinkB=0,
            ),
        )

        # Add speedup text
        plt.annotate(
            r'%.2f$\times$' % speedup,
            xy=((ada_x_at_sepcma_loss + sepcma_x_end)/2, sepcma_y_end),
            xytext=((ada_x_at_sepcma_loss + sepcma_x_end)/2, sepcma_y_end * 1.02),
            horizontalalignment='center',
            verticalalignment='bottom'
        )

        print(f"\nSpeedup: AdaSmoothES is {speedup:.2f}x faster than SepCMAES!")

    # Formatting
    plt.xlabel('Iterations ($T$)', fontsize=14)
    plt.ylabel('Optimality Gap (log scale)', fontsize=14)
    plt.title(f'{func_name.capitalize()} (d={dimension}, K={num_queries})', fontsize=16)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)

    # Save figure
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f'{func_name}_comparison_K{num_queries}.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {save_path}")

    plt.show()
    plt.close()

    # Print final losses
    print("\n" + "="*70)
    print("FINAL LOSSES")
    print("="*70)
    for opt, history in sorted(histories, key=lambda x: x[1][-1]):
        final_loss = history[-1]
        print(f"{opt['label']:<30} {final_loss:10.2f}")


def plot_multiple_functions(
    funcs: list = ['quadratic', 'levy', 'rosenbrock', 'ackley'],
    dimension: int = 1000,
    num_iterations: int = 10000,
    num_queries: int = 10,
    mu: float = 0.05,
    seed: int = 456,
    save_dir: str = 'figures'
):
    """Plot comparison for multiple test functions in a single figure."""

    # Optimizer configurations
    optimizers = [
        {'name': 'vanilla', 'label': 'Vanilla'},
        {'name': 'zoar', 'label': 'ZoAR'},
        {'name': 'sepcmaes', 'label': 'SepCMAES'},
        {'name': 'adasmooth_es', 'label': 'AdaSmoothES', 'config_suffix': '_adaptive_betacma_match_cma_decay0.001'},
    ]

    markers = [r'$\heartsuit$', r'$\triangle$', r'$\circ$', r'$\star$']
    colors = ['#1E90FF', '#FF6347', '#D2691E', '#9400D3']
    n = 16

    plt.figure(figsize=(14, 3.5))

    for func_idx, func_name in enumerate(funcs):
        plt.subplot(1, len(funcs), func_idx + 1)

        for i, opt in enumerate(optimizers):
            try:
                config_suffix = opt.get('config_suffix', '')
                filename = f"{func_name}_{opt['name']}{config_suffix}_d{dimension}_ni{num_iterations}_nq{num_queries}_mu{mu}_s{seed}.pt"
                filepath = Path('results/synthetic_comparison') / filename

                if not filepath.exists():
                    print(f"Warning: File not found: {filepath}")
                    continue

                history = torch.load(filepath, weights_only=True)

                start = 0
                end = len(history)
                interval = max((end - start) // n, 1)
                xs = torch.arange(start, end, interval)
                ys = torch.log(torch.tensor(history))[start:end:interval]

                plt.plot(xs, ys, marker=markers[i], color=colors[i],
                        label=opt['label'], linestyle='--', markersize=6.0)

            except Exception as e:
                print(f"Error loading {opt['name']} for {func_name}: {e}")
                continue

        plt.xlabel('Iters ($T$)', fontsize=12)
        plt.title(func_name.capitalize(), fontsize=14)

        if func_idx == 0:
            plt.ylabel("Optimality Gap (log scale)", fontsize=12)
        if func_idx == len(funcs) - 1:
            plt.legend(fontsize=9, loc='upper right')

    plt.tight_layout()

    # Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f'multi_function_comparison_K{num_queries}.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nMulti-function figure saved to: {save_path}")

    plt.show()
    plt.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Plot AdaSmoothES Comparison')
    parser.add_argument('--func', default='rosenbrock', help='Test function (or "multi" for all)')
    parser.add_argument('--dim', type=int, default=1000, help='Dimension')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--queries', type=int, default=10, help='Number of queries')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--save-dir', default='figures', help='Save directory')

    args = parser.parse_args()

    if args.func == 'multi':
        print("Plotting multiple functions...")
        plot_multiple_functions(
            funcs=['quadratic', 'levy', 'rosenbrock', 'ackley'],
            dimension=args.dim,
            num_iterations=args.iters,
            num_queries=args.queries,
            seed=args.seed,
            save_dir=args.save_dir
        )
    else:
        print(f"Plotting single function: {args.func}...")
        plot_comparison(
            func_name=args.func,
            dimension=args.dim,
            num_iterations=args.iters,
            num_queries=args.queries,
            seed=args.seed,
            save_dir=args.save_dir
        )


if __name__ == '__main__':
    main()
