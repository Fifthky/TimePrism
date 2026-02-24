#!/bin/bash

declare -a datasets=('electricity' 'exchange' 'solar' 'traffic' 'wiki')
num_hyp_ckpt=1
CKPT_JSON='ckpts.json'
CKPT_PATHS=$(cat ${CKPT_JSON})

# --- Flexible Argument Parsing Block (This part is correct and remains unchanged) ---

# Step 1: Check for required arguments
if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 <seed> <dataset> <model> [optional_args...]"
    echo "Example: $0 3141 solar timeMCL 16 relaxed-wta mean Short"
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
scaler_type_override=""

# Step 4: Loop through all optional arguments (from 4th position) to identify them by content
for arg in "${@:4}"; do
    case $arg in
        Short|Full)
            history_mode=$arg
            ;;
        relaxed-wta|awta)
            wta_mode=$arg
            ;;
        mean|mean_std)
            scaler_type_override=$arg
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
echo "--- Parsed Settings for Evaluation ---"
echo "Seed: $seed"
echo "Dataset(s): $datasets"
echo "Model: $model"
echo "Num Hypotheses: $num_hyp"
echo "WTA Mode: $wta_mode"
echo "Scaler Type Override: ${scaler_type_override:-'Using Defaults'}"
echo "History Mode: $history_mode (Hydra override: ${history_override})"
echo "--------------------------------------"
# --- END of Argument Parsing Block ---


if [ $seed == "all_seeds" ]; then
    seeds=('3141' '3142' '3143')
else
    seeds=($seed)
fi

if [ $datasets == "all" ]; then
    datasets=('electricity' 'exchange' 'solar' 'traffic' 'wiki')
elif [ $datasets == "all_fev" ]; then
    datasets=('uci_air' 'hospital' 'mdense' 'hierachi')
else
    datasets=($datasets)
fi

if [ -n "$scaler_type_override" ]; then
    echo "Using provided scaler_type: $scaler_type_override"
    declare -A scaler_type_dict
    scaler_type_dict["electricity"]=$scaler_type_override
    scaler_type_dict["exchange"]=$scaler_type_override
    scaler_type_dict["solar"]=$scaler_type_override
    scaler_type_dict["traffic"]=$scaler_type_override
    scaler_type_dict["wiki"]=$scaler_type_override
    scaler_type_dict["uci_air"]=$scaler_type_override
    scaler_type_dict["hospital"]=$scaler_type_override
    scaler_type_dict["mdense"]=$scaler_type_override
    scaler_type_dict["hierachi"]=$scaler_type_override
else
    echo "No scaler_type provided, setting to the default scaler_type"
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

for dataset in "${datasets[@]}"; do
    max_epochs=${max_epochs_dict[$dataset]}

    scaler_type=${scaler_type_dict[$dataset]}
    batch_size=200 
    
    for seed in "${seeds[@]}"; do

        if [ $model == "tactis2" ]; then
            base_key="seed_${seed}_${dataset}_${model}_${num_hyp}_hist_${history_mode}"
            key_phase1="${base_key}_phase_1"
            key_phase2="${base_key}_phase_2"
            ckpt_path_phase1=$(echo ${CKPT_PATHS} | jq -r --arg key "${key_phase1}" '.[$key]')
            ckpt_path_phase2=$(echo ${CKPT_PATHS} | jq -r --arg key "${key_phase2}" '.[$key]')

            if [ -n "${ckpt_path_phase1}" ] && [ "${ckpt_path_phase1}" != "null" ]; then
                echo "Evaluating ${ckpt_path_phase1}" and ${ckpt_path_phase2}
                python train.py data=${dataset}.yaml ckpt_path_phase1=${ckpt_path_phase1} ckpt_path_phase2=${ckpt_path_phase2} experiment=${dataset}.yaml model=${model}.yaml run_name=eval_${base_key} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True data.batch_size=${batch_size} ${history_override}
            else
                echo "ckpt_path is None for ${key_phase1}"
            fi
        
        elif [ $model == "timeMCL" ]; then
            if [ $wta_mode == "awta" ]; then
                key="seed_${seed}_${dataset}_${model}_${num_hyp}_awta_temp_ini_10_decay_0.95_scaler_${scaler_type}_hist_${history_mode}"
            elif [ $wta_mode == "relaxed-wta" ]; then
                key="seed_${seed}_${dataset}_${model}_${num_hyp}_relaxed-wta_epsilon_0.1_scaler_${scaler_type}_hist_${history_mode}"
            fi

            ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
            if [ -n "${ckpt_path}" ] && [ "${ckpt_path}" != "null" ]; then
                echo "Evaluating ${ckpt_path}"
                python train.py data=${dataset}.yaml ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=eval_${key} model.params.num_hypotheses=${num_hyp} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True model.compute_flops=False model.params.wta_mode=$wta_mode data.batch_size=${batch_size} model.params.scaler_type=${scaler_type} ${history_override}
            else
                echo "ckpt_path is None for ${key}"
            fi

        elif [ $model == "timePrism" ] || [ $model == "timePrism_iTran" ]; then
            key="seed_${seed}_${dataset}_${model}_${num_hyp}_hist_${history_mode}"
            
            ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
            
            if [ -n "${ckpt_path}" ] && [ "${ckpt_path}" != "null" ]; then
                echo "Evaluating ${ckpt_path} for model ${model}"
                python train.py data=${dataset}.yaml ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=eval_${key} logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=${seed} train=False test=True model.compute_flops=False data.batch_size=${batch_size} model.params.scaler_type=${scaler_type} ${history_override}
            else
                echo "ckpt_path is None for key: ${key}"
            fi
        
        else
            # It now correctly uses the parsed ${num_hyp} variable and adds the history mode.
            key="seed_${seed}_${dataset}_${model}_${num_hyp}_hist_${history_mode}"
            
            ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
            if [ -n "${ckpt_path}" ] && [ "${ckpt_path}" != "null" ]; then
                echo "Evaluating ${ckpt_path}"
                python train.py data=${dataset}.yaml ckpt_path=${ckpt_path} experiment=${dataset}.yaml model=${model}.yaml run_name=eval_${key} model.params.num_hypotheses=100 logger.mlflow.experiment_name=eval_${dataset}_${max_epochs} task_name=eval_${dataset}_${max_epochs} trainer.max_epochs=${max_epochs} seed=3141 train=False test=True model.compute_flops=False data.batch_size=${batch_size} model.params.scaler_type=${scaler_type} ${history_override}
            else
                echo "ckpt_path is None for ${key}"
            fi
        fi
    done
done