#!/bin/bash

# Default experiment configuration
# Override by setting environment variables before running this script
# Example: EXP=adversarial FUNC=ackley SEED=123 bash run_script.sh

# Experiment type: "synthetic" or "adversarial"
EXP=${EXP:-synthetic}

# Synthetic function options
FUNC=${FUNC:-levy}  # ackley, levy, rosenbrock, quadratic, rastrigin

# Adversarial attack options
DATASET=${DATASET:-mnist}  # mnist, cifar10
IDX=${IDX:-0}  # image index to attack
DEVICE=${DEVICE:-cuda}  # cuda, cpu, mps

# Optimizer configuration
OPTIMIZER=${OPTIMIZER:-zoar}  # fo, es, vanilla, rl, xnes, sepcmaes, adasmooth, twopoint, zoo, reinforce, zoar, zoar_0, relizo, zohs, zohs_expavg

# Optimization hyperparameters
SEED=${SEED:-456}
DIM=${DIM:-10000}
ITERATIONS=${ITERATIONS:-20000}
LR=${LR:-0.001}
UPDATE_RULE=${UPDATE_RULE:-radazo}  # sgd, adam, radazo

# ZO-specific parameters
NUM_QUERIES=${NUM_QUERIES:-10}
MU=${MU:-0.05}  # zo_eps
NUM_HISTORIES=${NUM_HISTORIES:-5}

# Baseline configuration (for zoo/reinforce)
BASELINE=${BASELINE:-single}  # single, average

# Generate tag for easy tracking
if [ "$EXP" == "synthetic" ]; then
    TAG=${TAG:-$FUNC-$OPTIMIZER-$UPDATE_RULE-d$DIM-$SEED}
else
    TAG=${TAG:-$DATASET-$OPTIMIZER-$UPDATE_RULE-idx$IDX-$SEED}
fi

# Print configuration
echo "======================================"
echo "ZoAR Experiment Runner"
echo "======================================"
echo "Experiment Type: $EXP"
if [ "$EXP" == "synthetic" ]; then
    echo "Function: $FUNC"
    echo "Dimension: $DIM"
    echo "Iterations: $ITERATIONS"
else
    echo "Dataset: $DATASET"
    echo "Image Index: $IDX"
    echo "Device: $DEVICE"
fi
echo "--------------------------------------"
echo "Optimizer: $OPTIMIZER"
echo "Update Rule: $UPDATE_RULE"
echo "Learning Rate: $LR"
echo "Num Queries: $NUM_QUERIES"
echo "Mu (ZO eps): $MU"
echo "Num Histories: $NUM_HISTORIES"
echo "Seed: $SEED"
echo "Tag: $TAG"
echo "======================================"

# Create temporary config file
TEMP_CONFIG="/tmp/zoar_temp_config_$$.yaml"

if [ "$EXP" == "synthetic" ]; then
    cat > $TEMP_CONFIG <<EOF
exp: synthetic

# experiment options
func_name: $FUNC
seed: $SEED
dimension: $DIM
num_iterations: $ITERATIONS
optimizers:
  - $OPTIMIZER

# optimization options
lr: $LR
betas:
  - 0.9
  - 0.99
epsilon: 1.0e-8
update_rule: $UPDATE_RULE

# ZO algorithm parameters
num_queries: $NUM_QUERIES
mu: $MU

# History parameters
num_histories: $NUM_HISTORIES

# Baseline configuration (for zoo/reinforce)
baseline: $BASELINE
EOF

elif [ "$EXP" == "adversarial" ]; then
    cat > $TEMP_CONFIG <<EOF
exp: adversarial

# experiment options
dataset: $DATASET
idx: $IDX
device: $DEVICE
seed: $SEED
num_iterations: $ITERATIONS
optimizers:
  - $OPTIMIZER

# optimization options
lr: $LR
betas:
  - 0.9
  - 0.99
epsilon: 1.0e-8
update_rule: $UPDATE_RULE

# ZO algorithm parameters
num_queries: $NUM_QUERIES
mu: $MU

# History parameters
num_histories: $NUM_HISTORIES

# Baseline configuration (for zoo/reinforce)
baseline: $BASELINE
EOF

else
    echo "Error: Unknown experiment type '$EXP'. Use 'synthetic' or 'adversarial'."
    exit 1
fi

echo "Running experiment..."
echo ""

# Run the experiment
python run.py --config $TEMP_CONFIG "$@"

# Save exit code
EXIT_CODE=$?

# Clean up temporary config
rm -f $TEMP_CONFIG

# Exit with the same code as the Python script
exit $EXIT_CODE
