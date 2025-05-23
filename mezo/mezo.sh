OPT=${OPT:-zoar} # vanilla, zoar
TASK=${TASK:-SST2}
OPT_TYPE=${OPT_TYPE:-radazo} # sgd, adam, radazo
NUM_HISTORIES=${NUM_HISTORIES:-15}

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

# BS=${BS:-32}
BS=${BS:-16}
LR=${LR:-5e-5}
EPS=${EPS:-1e-2}
TRAIN_SET_SEED=${TRAIN_SET_SEED:-0}
SEED=${SEED:-456}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-500}

MODE=${MODE:-lora}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=mezo-$MODEL_NAME-$OPT-$TASK-$MODE-$STEPS-$BS-$LR-$EPS-$OPT_TYPE-$NUM_HISTORIES-s$SEED

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN SET SEED: $TRAIN_SET_SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "OPT: $OPT"
echo "OPT_TYPE: $OPT_TYPE"
echo "NUM_HISTORIES: $NUM_HISTORIES"

python run.py \
    --opt_name $OPT \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $TRAIN_SET_SEED --seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --max_steps $STEPS \
    --trainer zo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS \
    --train_as_classification \
    --overwrite_output_dir \
    --report_to wandb \
    --run_name $TAG \
    --logging_first_step true \
    --logging_steps 1 \
    --logging_strategy steps \
    --opt_type $OPT_TYPE \
    --num_histories $NUM_HISTORIES \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"