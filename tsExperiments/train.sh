#!/bin/bash
declare -a num_hyps=('1')
scaler_type="mean"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Flexible Argument Parsing Block ---

# Step 1: Check for required arguments
if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 <seed> <dataset> <model> [optional_args...]"
    echo "Example: $0 3141 solar timeMCL 16 relaxed-wta Short"
    echo "Example (skipping args): $0 3141 solar ETS Short"
    exit 1
fi

# Step 2: Assign required positional arguments
seed=$1
datasets=$2
model=$3

# Step 3: Set defaults for all optional arguments
num_hyp=1
wta_mode="None"
history_mode="Full"

# Step 4: Loop through all optional arguments (from 4th position) to identify them by content
for arg in "${@:4}"; do
    case $arg in
        Short|Full)
            history_mode=$arg
            ;;
        relaxed-wta|awta)
            wta_mode=$arg
            ;;
        # Regex to check if the argument is a positive integer
        ''|*[!0-9]*)
            echo "Warning: Unrecognized non-numeric argument '$arg'. Ignoring."
            ;;
        *)
            # If it's a number, assume it's num_hyp
            num_hyp=$arg
            ;;
    esac
done

# Step 5: Convert descriptive history_mode to a boolean flag for Hydra
if [ "$history_mode" == "Full" ]; then
    use_full_history_flag="True"
else
    use_full_history_flag="False"
fi
history_override="+model.params.use_full_history=${use_full_history_flag}"

# --- Log the parsed settings for user confirmation ---
echo "--- Parsed Settings ---"
echo "Seed: $seed"
echo "Dataset(s): $datasets"
echo "Model: $model"
echo "Num Hypotheses: $num_hyp"
echo "WTA Mode: $wta_mode"
echo "History Mode: $history_mode (Hydra override: ${history_override})"
echo "-----------------------"
# --- END of Argument Parsing Block ---


if [ $datasets == "all" ]; then
    datasets=('electricity' 'exchange' 'solar' 'traffic' 'wiki')
else
    datasets=($datasets)
fi

declare -A max_epochs_dict
max_epochs_dict["electricity"]=200
max_epochs_dict["exchange"]=200
max_epochs_dict["solar"]=200
max_epochs_dict["traffic"]=200
max_epochs_dict["wiki"]=200
max_epochs_dict["uci_air"]=200
max_epochs_dict["hospital"]=200
max_epochs_dict["mdense"]=200
max_epochs_dict["hierachi"]=200

declare -A scaler_type_dict
scaler_type_dict["electricity"]="mean"
scaler_type_dict["exchange"]="mean"
scaler_type_dict["solar"]="mean"
scaler_type_dict["traffic"]="mean"
scaler_type_dict["wiki"]="mean"
scaler_type_dict["uci_air"]="mean"
scaler_type_dict["hospital"]="mean"
scaler_type_dict["mdense"]="mean"
scaler_type_dict["hierachi"]="mean"

if [ $model == "timePrism" ] || [ $model == "timePrism_iTran" ]; then
    scaler_type_dict["exchange"]="mean_std" 
fi

if [ $model == "timeMCL" ]; then
    if [ $wta_mode == "awta" ]; then

        wta_mode_params_epsilon=0
        wta_mode_params_temperature_ini=10
        wta_mode_params_scheduler_mode="exponential"
        wta_mode_params_temperature_decay=0.95
        wta_mode_params_temperature_lim=5e-4
        wta_mode_params_wta_after_temperature_lim=True

        for dataset in "${datasets[@]}"; do
            max_epochs=${max_epochs_dict[$dataset]}
            python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_${wta_mode}_temp_ini_${wta_mode_params_temperature_ini}_decay_${wta_mode_params_temperature_decay}_scaler_${scaler_type}_hist_${history_mode} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=${dataset}_${max_epochs} task_name=${dataset}_${max_epochs} model.params.wta_mode=${wta_mode} model.params.wta_mode_params.epsilon=${wta_mode_params_epsilon} model.params.wta_mode_params.temperature_ini=${wta_mode_params_temperature_ini} model.params.wta_mode_params.scheduler_mode=${wta_mode_params_scheduler_mode} model.params.wta_mode_params.temperature_decay=${wta_mode_params_temperature_decay} model.params.wta_mode_params.temperature_lim=${wta_mode_params_temperature_lim} model.params.wta_mode_params.wta_after_temperature_lim=${wta_mode_params_wta_after_temperature_lim} model.params.scaler_type=${scaler_type} trainer.max_epochs=${max_epochs} test=False model.params.scaler_type=${scaler_type_dict[$dataset]} seed=${seed} ${history_override}
        done

    elif [ $wta_mode == "relaxed-wta" ]; then

        wta_mode_params_epsilon=0.1

        for dataset in "${datasets[@]}"; do
            max_epochs=${max_epochs_dict[$dataset]}
            python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_${wta_mode}_epsilon_${wta_mode_params_epsilon}_scaler_${scaler_type}_hist_${history_mode} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=${dataset}_${max_epochs} task_name=${dataset}_${max_epochs} model.params.wta_mode=${wta_mode} model.params.wta_mode_params.epsilon=${wta_mode_params_epsilon} model.params.scaler_type=${scaler_type} trainer.max_epochs=${max_epochs} test=False seed=${seed} ${history_override}
        done

    else
        echo "WTA mode not specified or not applicable for ${model}. Running generic training."
        for dataset in "${datasets[@]}"; do
            max_epochs=${max_epochs_dict[$dataset]}
            python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_hist_${history_mode} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=train_${dataset}_${max_epochs} task_name=${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=${seed} test=False model.params.scaler_type=${scaler_type_dict[$dataset]} ${history_override}
        done
    fi
elif [ $model == "ETS" ]; then
    for dataset in "${datasets[@]}"; do
        max_epochs=${max_epochs_dict[$dataset]}
        python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_hist_${history_mode} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=${seed} test=True ${history_override}
    done
elif [ $model == "tactis2" ]; then
    for dataset in "${datasets[@]}"; do
        max_epochs=${max_epochs_dict[$dataset]}
        python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_hist_${history_mode} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=train_${dataset}_${max_epochs} task_name=${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=${seed} test=False ${history_override}
    done
else
    for dataset in "${datasets[@]}"; do
        max_epochs=${max_epochs_dict[$dataset]}
        python train.py data=${dataset}.yaml experiment=${dataset}.yaml model=${model}.yaml run_name=seed_${seed}_${dataset}_${model}_${num_hyp}_hist_${history_mode} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=train_${dataset}_${max_epochs} task_name=${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=${seed} test=False model.params.scaler_type=${scaler_type_dict[$dataset]} ${history_override}
    done
fi