#!/bin/bash

# --- Configuration ---
# Define the dataset and epoch count. Note: these settings are for a compute/FLOPs run.
dataset='exchange'
max_epochs=1

# Set default values for model-specific parameters
wta_mode='relaxed-wta'
scaler_type='mean'

cd ..

# Determine mode and seed from script arguments.
# Supported calls:
#   ./flops.sh                        -> default seed, all models, original logic
#   ./flops.sh 1                      -> seed=1, all models, original logic
#   ./flops.sh --timePrism_n          -> TimePrism-only N-sweep (special logic), default seed
#   ./flops.sh 1 --timePrism_n        -> TimePrism-only N-sweep, seed=1
mode=""
default_seed=3142

if [ "$1" == "--timePrism_n" ]; then
    mode="timePrism_n"
    seed=${default_seed}
elif [ "$2" == "--timePrism_n" ]; then
    mode="timePrism_n"
    seed=$1
elif [ $# -ge 1 ]; then
    seed=$1
else
    seed=${default_seed}
fi

echo "Using seed: ${seed}"
if [ -n "$mode" ]; then
    echo "Special FLOPs mode: ${mode}"
fi

# --- Model Execution ---
# The script executes each model with a fixed hypothesis count for architecture definition.
# A loop is added to iterate over different numbers of parallel samples.

if [ "$mode" == "timePrism_n" ]; then
    # Special logic: TimePrism-only FLOPs sweep over different N, with num_parallel_samples fixed to 1.
    declare -a N_list=(1 16 256 625 1024)
    num_parallel_samples=1

    echo "====================================================="
    echo "--- Running TimePrism FLOPs sweep over N (num_parallel_samples=${num_parallel_samples}) ---"
    echo "====================================================="

    for N in "${N_list[@]}"; do
        model='timePrism'
        echo "--- Running: ${model} with N=${N} ---"
        python train.py experiment=${dataset}.yaml model=${model}.yaml \
            run_name=seed_${seed}_${dataset}_${model}_${N}_hist_Short_compute_ps_${num_parallel_samples} \
            model.params.num_hypotheses=${N} \
            model.params.num_parallel_samples=${num_parallel_samples} \
            +model.params.use_full_history=False \
            logger.mlflow.experiment_name=compute_${dataset}_${max_epochs}_timePrismN \
            task_name=compute_${dataset} \
            trainer.max_epochs=${max_epochs} \
            seed=${seed} \
            train=True test=False model.compute_flops=True \
            trainer.validation_only=True \
            data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1
    done

    # After collecting FLOPs for all N, extract a simple n-vs-flops CSV.
    python computation_flops/extract_flops.py \
        --experiment compute_${dataset}_${max_epochs}_timePrismN \
        --output ../computation_flops/flops_summary_timePrism.csv \
        --timePrism_n
else
    # Default logic: test multiple models and multiple parallel sample counts.

    # Define the number of parallel samples to test
    declare -a parallel_samples_list=(1 10 100)

    for num_parallel_samples in "${parallel_samples_list[@]}"; do
        echo "====================================================="
        echo "--- Running models with num_parallel_samples=${num_parallel_samples} ---"
        echo "====================================================="

        # 1. Tactis
        model='tactis2'
        echo "--- Running: ${model} ---"
        # --- FIX: tactis2 has a two-phase training and requires at least nb_epoch_phase_1 (20) epochs to run. ---
        # --- We set max_epochs to 20 specifically for this model to satisfy its minimum requirement. ---
        python train.py experiment=${dataset}.yaml model=${model}.yaml \
            run_name=seed_${seed}_${dataset}_${model}_1_hist_Full_compute_ps_${num_parallel_samples} \
            model.params.num_hypotheses=1 \
            model.params.num_parallel_samples=${num_parallel_samples} \
            +model.params.use_full_history=True \
            logger.mlflow.experiment_name=compute_${dataset}_${max_epochs} \
            task_name=compute_${dataset} \
            trainer.max_epochs=20 \
            seed=${seed} \
            train=True test=False model.compute_flops=True \
            data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1

        # 2. TempFlow
        model='tempflow'
        echo "--- Running: ${model} ---"
        python train.py experiment=${dataset}.yaml model=${model}.yaml \
            run_name=seed_${seed}_${dataset}_${model}_1_hist_Full_compute_ps_${num_parallel_samples} \
            model.params.num_hypotheses=1 \
            model.params.num_parallel_samples=${num_parallel_samples} \
            +model.params.use_full_history=True \
            logger.mlflow.experiment_name=compute_${dataset}_${max_epochs} \
            task_name=compute_${dataset} \
            trainer.max_epochs=${max_epochs} \
            seed=${seed} \
            train=True test=False model.compute_flops=True \
            data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1

        # 3. DeepAR
        model='deepAR'
        echo "--- Running: ${model} ---"
        python train.py experiment=${dataset}.yaml model=${model}.yaml \
            run_name=seed_${seed}_${dataset}_${model}_1_hist_Full_compute_ps_${num_parallel_samples} \
            model.params.num_hypotheses=1 \
            model.params.num_parallel_samples=${num_parallel_samples} \
            +model.params.use_full_history=True \
            logger.mlflow.experiment_name=compute_${dataset}_${max_epochs} \
            task_name=compute_${dataset} \
            trainer.max_epochs=${max_epochs} \
            seed=${seed} \
            train=True test=False model.compute_flops=True \
            trainer.validation_only=True \
            data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1

        # 4. timeMCL
        model='timeMCL'
        echo "--- Running: ${model} ---"
        # --- FIX: Added the missing 'epsilon' parameter required by the 'relaxed-wta' loss mode. ---
        python train.py experiment=${dataset}.yaml model=${model}.yaml \
            run_name=seed_${seed}_${dataset}_${model}_16_hist_Full_${wta_mode}_scaler_${scaler_type}_compute_ps_${num_parallel_samples} \
            model.params.num_hypotheses=16 \
            model.params.num_parallel_samples=${num_parallel_samples} \
            +model.params.use_full_history=True \
            logger.mlflow.experiment_name=compute_${dataset}_${max_epochs} \
            task_name=compute_${dataset} \
            model.params.wta_mode=${wta_mode} \
            model.params.wta_mode_params.epsilon=0.1 \
            model.params.scaler_type=${scaler_type} \
            trainer.max_epochs=${max_epochs} \
            seed=${seed} \
            train=True test=False model.compute_flops=True \
            trainer.validation_only=True \
            data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1

        # 5. TimeGrad
        model='timeGrad'
        echo "--- Running: ${model} ---"
        python train.py experiment=${dataset}.yaml model=${model}.yaml \
            run_name=seed_${seed}_${dataset}_${model}_1_hist_Full_compute_ps_${num_parallel_samples} \
            model.params.num_hypotheses=1 \
            model.params.num_parallel_samples=${num_parallel_samples} \
            +model.params.use_full_history=True \
            logger.mlflow.experiment_name=compute_${dataset}_${max_epochs} \
            task_name=compute_${dataset} \
            trainer.max_epochs=${max_epochs} \
            seed=${seed} \
            train=True test=False model.compute_flops=True \
            trainer.validation_only=True \
            data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1

        # 6. TransformerTempFlow
        model='transformer_tempflow'
        echo "--- Running: ${model} ---"
        python train.py experiment=${dataset}.yaml model=${model}.yaml \
            run_name=seed_${seed}_${dataset}_${model}_1_hist_Full_compute_ps_${num_parallel_samples} \
            model.params.num_hypotheses=1 \
            model.params.num_parallel_samples=${num_parallel_samples} \
            +model.params.use_full_history=True \
            logger.mlflow.experiment_name=compute_${dataset}_${max_epochs} \
            task_name=compute_${dataset} \
            trainer.max_epochs=${max_epochs} \
            seed=${seed} \
            train=True test=False model.compute_flops=True \
            trainer.validation_only=True \
            data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1

        # 7. TimePrism
        model='timePrism'
        echo "--- Running: ${model} ---"
        python train.py experiment=${dataset}.yaml model=${model}.yaml \
            run_name=seed_${seed}_${dataset}_${model}_625_hist_Short_compute_ps_${num_parallel_samples} \
            model.params.num_hypotheses=625 \
            model.params.num_parallel_samples=${num_parallel_samples} \
            +model.params.use_full_history=False \
            logger.mlflow.experiment_name=compute_${dataset}_${max_epochs} \
            task_name=compute_${dataset} \
            trainer.max_epochs=${max_epochs} \
            seed=${seed} \
            train=True test=False model.compute_flops=True \
            trainer.validation_only=True \
            data.num_batches_per_epoch=1 data.num_batches_val_per_epoch=1

    done
fi