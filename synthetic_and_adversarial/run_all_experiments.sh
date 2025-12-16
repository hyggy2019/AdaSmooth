#!/bin/bash

# Run ALL experiments from TODO.md automatically
# This runs:
# 1. Synthetic: rosenbrock, ackley, rastrigin × d=1000, 5000, 10000
# 2. Adversarial: mnist, cifar10

echo "========================================================================"
echo "Running ALL TODO.md Experiments"
echo "========================================================================"
echo ""
echo "This will run:"
echo "  - 3 functions × 3 dimensions = 9 synthetic experiments"
echo "  - 2 datasets = 2 adversarial experiments"
echo "  - Total: 11 experiments"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Track time
START_TIME=$(date +%s)

# ========================================================================
# 1. Synthetic Functions
# ========================================================================

echo "========================================================================"
echo "PART 1: SYNTHETIC FUNCTIONS"
echo "========================================================================"

for FUNC in rosenbrock ackley rastrigin; do
    for DIM in 1000 5000 10000; do
        echo ""
        echo "--------------------------------------------------------------------"
        echo "Running: $FUNC, d=$DIM"
        echo "--------------------------------------------------------------------"
        bash run_all_todo.sh $FUNC $DIM synthetic

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed for $FUNC d=$DIM"
        else
            echo "SUCCESS: $FUNC d=$DIM completed"
        fi
    done
done

# ========================================================================
# 2. Adversarial Attacks
# ========================================================================

echo ""
echo "========================================================================"
echo "PART 2: ADVERSARIAL ATTACKS"
echo "========================================================================"

for DATASET in mnist cifar10; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Running: $DATASET"
    echo "--------------------------------------------------------------------"
    bash run_all_todo.sh $DATASET 1000 adversarial

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed for $DATASET"
    else
        echo "SUCCESS: $DATASET completed"
    fi
done

# ========================================================================
# Summary
# ========================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================================================"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Next steps:"
echo "  1. Check results in results/synthetic/ and results/attack/"
echo "  2. Run plotting: python plot_all_results.py"
echo "========================================================================"
