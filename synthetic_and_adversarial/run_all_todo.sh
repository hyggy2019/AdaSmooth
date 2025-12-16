#!/bin/bash

# Run all experiments as specified in TODO.md
# Usage: bash run_all_todo.sh [--function FUNC] [--dimension DIM] [--type TYPE]

FUNC="${1:-rosenbrock}"  # rosenbrock, ackley, rastrigin
DIM="${2:-1000}"         # 1000, 5000, 10000
TYPE="${3:-synthetic}"   # synthetic, adversarial

echo "============================================================"
echo "Running TODO.md Experiments"
echo "============================================================"
echo "Function: $FUNC"
echo "Dimension: $DIM"
echo "Type: $TYPE"
echo "============================================================"

if [ "$TYPE" = "synthetic" ]; then
    # Synthetic function experiments
    if [ "$DIM" = "1000" ]; then
        CONFIG="config/synthetic.yaml"
    elif [ "$DIM" = "5000" ]; then
        CONFIG="config/synthetic-d5000.yaml"
    elif [ "$DIM" = "10000" ]; then
        CONFIG="config/synthetic-d10000.yaml"
    else
        echo "Error: Invalid dimension $DIM"
        exit 1
    fi

    echo "Running: python run.py --config $CONFIG --func_name $FUNC"
    python run.py --config $CONFIG --func_name $FUNC

elif [ "$TYPE" = "adversarial" ]; then
    # Adversarial attack experiments
    if [ "$FUNC" = "mnist" ]; then
        CONFIG="config/adversarial.yaml"
    elif [ "$FUNC" = "cifar10" ]; then
        CONFIG="config/adversarial-cifar10.yaml"
    else
        echo "Error: Invalid dataset $FUNC (use mnist or cifar10)"
        exit 1
    fi

    echo "Running: python run.py --config $CONFIG"
    python run.py --config $CONFIG

else
    echo "Error: Invalid type $TYPE (use synthetic or adversarial)"
    exit 1
fi

echo "============================================================"
echo "Done!"
echo "============================================================"
