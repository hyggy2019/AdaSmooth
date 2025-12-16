"""
Plot comparison of different adaptive beta schedulers for AdaSmoothES.

Visualizes the performance of different divergence/scheduler combinations.

Author: AdaSmoothES divergence comparison
"""

import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
from pathlib import Path

# Set matplotlib style
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['mathtext.fontset'] = 'cm'


def plot_divergence_comparison(
    func_name: str = 'rosenbrock',
    dimension: int = 1000,
    num_iterations: int = 10000,
    num_queries: int = 10,
    mu: float = 0.05,
    seed: int = 456,
    save_dir: str = 'figures',
    results_dir: str = 'results/synthetic_divergences'
):
    """
    Plot optimization curves comparing different adaptive beta schedulers.
    """

    # Scheduler configurations with their file suffixes and labels
    configs = [
        {
            'suffix': '_adaptive_betacma_match_cma_decay0.001',
            'label': 'CMA Match + Decay',
            'marker': r'$\star$',
            'color': '#9400D3'
        },
        {
            'suffix': '_adaptive_betastd_c_beta1.0',
            'label': 'Adaptive Std (c=1.0)',
            'marker': r'$\circ$',
            'color': '#1E90FF'
        },
        {
            'suffix': '_adaptive_betaentropy_target_target_eff_ratio0.5',
            'label': 'Entropy Target (0.5)',
            'marker': r'$\triangle$',
            'color': '#3CB371'
        },
        {
            'suffix': '_adaptive_betaentropy_target_target_eff_ratio0.3',
            'label': 'Entropy Target (0.3)',
            'marker': r'$\boxdot$',
            'color': '#FF6347'
        },
        {
            'suffix': '_beta_decay0.001_beta_init10.0_beta_schedulepolynomial',
            'label': 'Fixed Polynomial',
            'marker': 'x',
            'color': '#D2691E'
        },
    ]

    n = 16  # Number of points to plot

    # Main figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # === Left plot: Full optimization curves ===
    histories = []
    final_losses = []

    for config in configs:
        try:
            # Load result file
            filename = f"{func_name}_adasmooth_es{config['suffix']}_d{dimension}_ni{num_iterations}_nq{num_queries}_mu{mu}_s{seed}.pt"
            filepath = Path(results_dir) / filename

            if not filepath.exists():
                print(f"Warning: File not found: {filepath}")
                continue

            history = torch.load(filepath, weights_only=True)
            histories.append((config, history))
            final_losses.append((config['label'], history[-1]))

            # Prepare data for plotting
            start = 0
            end = len(history)
            interval = max((end - start) // n, 1)
            xs = torch.arange(start, end, interval)
            ys = torch.log(torch.tensor(history))[start:end:interval]

            # Plot on left axis
            ax1.plot(xs, ys, marker=config['marker'], color=config['color'],
                    label=config['label'], linestyle='--', markersize=7.0, linewidth=2)

        except Exception as e:
            print(f"Error loading {config['label']}: {e}")
            continue

    # Format left plot
    ax1.set_xlabel('Iterations ($T$)', fontsize=14)
    ax1.set_ylabel('Optimality Gap (log scale)', fontsize=14)
    ax1.set_title(f'{func_name.capitalize()} - Adaptive Scheduler Comparison\n(d={dimension}, K={num_queries})', fontsize=15)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # === Right plot: Zoomed in on convergence phase ===
    for config, history in histories:
        start = len(history) // 2  # Start from halfway
        end = len(history)
        interval = max((end - start) // n, 1)
        xs = torch.arange(start, end, interval)
        ys = torch.log(torch.tensor(history))[start:end:interval]

        ax2.plot(xs, ys, marker=config['marker'], color=config['color'],
                label=config['label'], linestyle='--', markersize=7.0, linewidth=2)

    # Format right plot
    ax2.set_xlabel('Iterations ($T$)', fontsize=14)
    ax2.set_ylabel('Optimality Gap (log scale)', fontsize=14)
    ax2.set_title(f'Convergence Phase (Zoomed)\n(Last {num_iterations//2} iterations)', fontsize=15)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f'{func_name}_divergence_comparison_K{num_queries}.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n📊 Figure saved to: {save_path}")

    plt.show()
    plt.close()

    # === Print detailed results ===
    print("\n" + "="*80)
    print("ADAPTIVE BETA SCHEDULER COMPARISON RESULTS")
    print("="*80)
    print(f"{'Rank':<6} {'Scheduler':<30} {'Final Loss':>12} {'vs Best':>10}")
    print("-"*80)

    # Sort by final loss
    final_losses.sort(key=lambda x: x[1])
    best_loss = final_losses[0][1]

    for i, (label, loss) in enumerate(final_losses, 1):
        vs_best = f"{((loss / best_loss - 1) * 100):+.1f}%"
        marker = "🏆" if i == 1 else ("✅" if loss < best_loss * 1.05 else "")
        print(f"{i:<6} {label:<30} {loss:12.2f} {vs_best:>10} {marker}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    best_scheduler = final_losses[0][0]
    best_loss_val = final_losses[0][1]
    worst_loss = final_losses[-1][1]
    improvement = ((worst_loss - best_loss_val) / worst_loss) * 100

    print(f"✅ Best Scheduler: {best_scheduler}")
    print(f"   Final Loss: {best_loss_val:.2f}")
    print(f"\n📈 Improvement over Fixed Polynomial: {improvement:.1f}%")
    print(f"   (Fixed: {worst_loss:.2f} → Best: {best_loss_val:.2f})")

    # Check which schedulers are within 5% of best
    good_schedulers = [label for label, loss in final_losses if loss < best_loss_val * 1.05]
    print(f"\n✅ Schedulers within 5% of best ({len(good_schedulers)}/{len(final_losses)}):")
    for scheduler in good_schedulers:
        print(f"   - {scheduler}")

    return final_losses


def plot_scheduler_bar_chart(
    func_name: str = 'rosenbrock',
    dimension: int = 1000,
    num_iterations: int = 10000,
    num_queries: int = 10,
    mu: float = 0.05,
    seed: int = 456,
    save_dir: str = 'figures',
    results_dir: str = 'results/synthetic_divergences'
):
    """Plot bar chart comparing final losses of different schedulers."""

    configs = [
        ('CMA Match + Decay', '_adaptive_betacma_match_cma_decay0.001'),
        ('Adaptive Std (c=1.0)', '_adaptive_betastd_c_beta1.0'),
        ('Entropy Target (0.5)', '_adaptive_betaentropy_target_target_eff_ratio0.5'),
        ('Entropy Target (0.3)', '_adaptive_betaentropy_target_target_eff_ratio0.3'),
        ('Fixed Polynomial', '_beta_decay0.001_beta_init10.0_beta_schedulepolynomial'),
    ]

    labels = []
    losses = []
    colors = ['#9400D3', '#1E90FF', '#3CB371', '#FF6347', '#D2691E']

    for label, suffix in configs:
        try:
            filename = f"{func_name}_adasmooth_es{suffix}_d{dimension}_ni{num_iterations}_nq{num_queries}_mu{mu}_s{seed}.pt"
            filepath = Path(results_dir) / filename

            if filepath.exists():
                history = torch.load(filepath, weights_only=True)
                labels.append(label)
                losses.append(history[-1])
            else:
                print(f"Warning: File not found: {filepath}")

        except Exception as e:
            print(f"Error loading {label}: {e}")
            continue

    if not losses:
        print("No data to plot!")
        return

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(labels)), losses, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, losses)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Formatting
    plt.xlabel('Adaptive Beta Scheduler', fontsize=14)
    plt.ylabel('Final Loss', fontsize=14)
    plt.title(f'{func_name.capitalize()} - Scheduler Performance Comparison\n(d={dimension}, K={num_queries}, {num_iterations} iterations)', fontsize=15)
    plt.xticks(range(len(labels)), labels, rotation=15, ha='right', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f'{func_name}_scheduler_bar_chart_K{num_queries}.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n📊 Bar chart saved to: {save_path}")

    plt.show()
    plt.close()

    # Print statistics
    best_loss = min(losses)
    best_idx = losses.index(best_loss)
    print(f"\n🏆 Best Scheduler: {labels[best_idx]} ({best_loss:.2f})")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Plot Divergence/Scheduler Comparison')
    parser.add_argument('--func', default='rosenbrock', help='Test function')
    parser.add_argument('--dim', type=int, default=1000, help='Dimension')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--queries', type=int, default=10, help='Number of queries')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--save-dir', default='figures', help='Save directory')
    parser.add_argument('--results-dir', default='results/synthetic_divergences', help='Results directory')
    parser.add_argument('--plot-type', choices=['curves', 'bar', 'both'], default='both',
                       help='Type of plot to generate')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ADAPTIVE BETA SCHEDULER VISUALIZATION")
    print("="*80)

    if args.plot_type in ['curves', 'both']:
        print("\nGenerating optimization curves...")
        plot_divergence_comparison(
            func_name=args.func,
            dimension=args.dim,
            num_iterations=args.iters,
            num_queries=args.queries,
            seed=args.seed,
            save_dir=args.save_dir,
            results_dir=args.results_dir
        )

    if args.plot_type in ['bar', 'both']:
        print("\nGenerating bar chart...")
        plot_scheduler_bar_chart(
            func_name=args.func,
            dimension=args.dim,
            num_iterations=args.iters,
            num_queries=args.queries,
            seed=args.seed,
            save_dir=args.save_dir,
            results_dir=args.results_dir
        )

    print("\n✅ All plots generated successfully!")


if __name__ == '__main__':
    main()
