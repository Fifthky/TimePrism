# ==============================================================================
# train_all.sh: A script to run multiple machine learning training jobs in parallel.
#
# Features:
# 1. Iterates over specified lists of seeds, datasets, and models.
# 2. Handles special command format for selected model.
# 3. Controls the maximum number of concurrent jobs to avoid overloading the server.
# 4. Redirects the output of each job to a separate log file for easy debugging.
# 5. Allows specifying visible GPUs via a command-line argument.
# 6. Gracefully terminates all child processes on exit (e.g., via Ctrl+C).
# ==============================================================================

# --- Configuration Section (Modify as needed) ---

# Set the maximum number of parallel jobs.
# Recommendation: Set this based on the number of GPUs or CPU cores on your server.
# For example, if you have 8 GPUs and each job uses 1, you can set this to 8.
MAX_JOBS=1

# Define the parameter lists to iterate over.
# SEEDS=(3141, 3142, 3143)
# DATASETS=("electricity" "exchange" "solar" "traffic" "wiki")
# DATASETS=("uci_air" "hospital" "mdense" "hierachi")
# MODELS=("timeGrad" "deepAR" "tempflow" "transformer_tempflow" "tactis2" "timeMCL" "timePrism")

SEEDS=(3141)
DATASETS=("solar")
MODELS=( "timePrism")

# Set the directory for log files.
LOG_DIR="logs/train_all"

# --- Graceful Shutdown Setup ---
# Define a cleanup function to be called when the script receives an interrupt signal.
cleanup() {
    echo "" # Add a newline for cleaner output
    echo ">>> Interrupt signal received. Terminating all child processes..."

    pkill -P $$
    
    echo ">>> All child processes terminated. Exiting."
    exit 1
}

# Set the trap to call the cleanup function upon receiving SIGINT (Ctrl+C).
trap cleanup SIGINT

# --- GPU Visibility Setup ---
# Check if a command-line argument is provided for GPU visibility.
# If so, set the CUDA_VISIBLE_DEVICES environment variable for this script and its children.
# This does not affect the parent shell environment from which this script is called.
# Special run modes (hyper, hyper_eval, run_16, run_16_eval) are treated as modes,
# not GPU identifiers, so they are excluded here.
if [ -n "$1" ] && [ "$1" != "hyper" ] && [ "$1" != "hyper_eval" ] && [ "$1" != "run_16" ] && [ "$1" != "run_16_eval" ]; then
    export CUDA_VISIBLE_DEVICES=$1
    echo "GPU visibility is set. Running jobs on specified GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "GPU visibility not specified. Using all available GPUs visible to the shell."
fi

# --- Hyper-parameter Training Function ---
# When called, this function launches hyper-parameter training jobs instead of the default grid.
run_hyperparam_training() {
    local SEEDS_HYPER=(3143)
    local DATASETS_HYPER=("solar" "exchange" "electricity")
    local NS=(1 256 1024)

    echo "--- Hyper-parameter training mode ---"

    for seed in "${SEEDS_HYPER[@]}"; do
        for dataset in "${DATASETS_HYPER[@]}"; do
            for N in "${NS[@]}"; do

                # Respect the same maximum parallel job limit as the main training loop.
                while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
                    echo "Reached max jobs ($MAX_JOBS) in hyper mode, waiting for a job to finish..."
                    wait -n
                done

                # Hyper-parameter training command (fixed form).
                COMMAND="bash train.sh $seed $dataset timePrism $N Short"

                # Include N in the log file name for easier debugging.
                LOG_FILE="${LOG_DIR}/hyper_seed-${seed}_data-${dataset}_model-timePrism_N-${N}.log"

                echo "Starting (hyper): $COMMAND"
                echo "Log -> $LOG_FILE"

                $COMMAND > "$LOG_FILE" 2>&1 &

                # A short pause to prevent a sudden burst of jobs that might lag the system.
                sleep 1
            done
        done
    done
}

# --- Hyper-parameter Eval Function ---
# When called, this function launches hyper-parameter evaluation jobs instead of training.
run_hyperparam_eval() {
    local SEEDS_HYPER=(3141 3142 3143)
    local DATASETS_HYPER=("solar" "exchange" "electricity")
    local NS=(1 16 256 625 1024)

    echo "--- Hyper-parameter eval mode ---"

    for seed in "${SEEDS_HYPER[@]}"; do
        for dataset in "${DATASETS_HYPER[@]}"; do
            for N in "${NS[@]}"; do

                # Respect the same maximum parallel job limit as the main training loop.
                while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
                    echo "Reached max jobs ($MAX_JOBS) in hyper_eval mode, waiting for a job to finish..."
                    wait -n
                done

                # Hyper-parameter eval command (fixed form).
                COMMAND="bash eval.sh $seed $dataset timePrism $N Short"

                # Include N in the log file name for easier debugging.
                LOG_FILE="${LOG_DIR}/hyper_eval_seed-${seed}_data-${dataset}_model-timePrism_N-${N}.log"

                echo "Starting (hyper_eval): $COMMAND"
                echo "Log -> $LOG_FILE"

                $COMMAND > "$LOG_FILE" 2>&1 &

                # A short pause to prevent a sudden burst of jobs that might lag the system.
                sleep 1
            done
        done
    done
}

# --- run_16 Training Function ---
# When called, this function launches training jobs for N=16 across multiple seeds and datasets.
run_16() {
    local SEEDS_RUN16=(3141 3142 3143)
    local DATASETS_RUN16=("electricity" "exchange" "solar" "traffic" "wiki")

    echo "--- run_16 training mode (N=16, timePrism) ---"

    for seed in "${SEEDS_RUN16[@]}"; do
        for dataset in "${DATASETS_RUN16[@]}"; do

            # Respect the same maximum parallel job limit as the main training loop.
            while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
                echo "Reached max jobs ($MAX_JOBS) in run_16 mode, waiting for a job to finish..."
                wait -n
            done

            COMMAND="bash train.sh $seed $dataset timePrism 16 Short"

            LOG_FILE="${LOG_DIR}/run16_seed-${seed}_data-${dataset}_model-timePrism_N-16.log"

            echo "Starting (run_16): $COMMAND"
            echo "Log -> $LOG_FILE"

            $COMMAND > "$LOG_FILE" 2>&1 &

            # A short pause to prevent a sudden burst of jobs that might lag the system.
            sleep 1

        done
    done
}

# --- run_16 Eval Function ---
# When called, this function launches evaluation jobs for N=16 across multiple seeds and datasets.
run_16_eval() {
    local SEEDS_RUN16=(3141 3142 3143)
    local DATASETS_RUN16=("electricity" "exchange" "solar" "traffic" "wiki")

    echo "--- run_16 eval mode (N=16, timePrism) ---"

    for seed in "${SEEDS_RUN16[@]}"; do
        for dataset in "${DATASETS_RUN16[@]}"; do

            # Respect the same maximum parallel job limit as the main training loop.
            while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
                echo "Reached max jobs ($MAX_JOBS) in run_16_eval mode, waiting for a job to finish..."
                wait -n
            done

            COMMAND="bash eval.sh $seed $dataset timePrism 16 Short"

            LOG_FILE="${LOG_DIR}/run16_eval_seed-${seed}_data-${dataset}_model-timePrism_N-16.log"

            echo "Starting (run_16_eval): $COMMAND"
            echo "Log -> $LOG_FILE"

            $COMMAND > "$LOG_FILE" 2>&1 &

            # A short pause to prevent a sudden burst of jobs that might lag the system.
            sleep 1

        done
    done
}

# --- Main Script Body ---

# Create the log directory if it doesn't exist.
mkdir -p "$LOG_DIR"
echo "--- Starting batch training jobs ---"
echo "Log files will be saved in: $LOG_DIR"
echo "Maximum concurrent jobs: $MAX_JOBS"
echo "--------------------------"

if [ "$1" == "hyper" ] || [ "$2" == "hyper" ]; then
    # Hyper-parameter training mode: do not enter the original logic.
    run_hyperparam_training
elif [ "$1" == "hyper_eval" ] || [ "$2" == "hyper_eval" ]; then
    # Hyper-parameter eval mode: do not enter the original logic.
    run_hyperparam_eval
elif [ "$1" == "run_16" ] || [ "$2" == "run_16" ]; then
    # N=16 training mode across multiple seeds and datasets.
    run_16
elif [ "$1" == "run_16_eval" ] || [ "$2" == "run_16_eval" ]; then
    # N=16 eval mode across multiple seeds and datasets.
    run_16_eval
else
    # Main loop to iterate through all parameter combinations (original logic).
    for seed in "${SEEDS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for model in "${MODELS[@]}"; do

                # 1. Check the current number of running background jobs.
                # `jobs -p` lists the Process IDs (PIDs) of all background jobs.
                # `wc -l` counts the number of lines, which corresponds to the number of jobs.
                while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
                    echo "Reached max jobs ($MAX_JOBS), waiting for a job to finish..."
                    # `wait -n` waits for any single background job to finish before proceeding.
                    wait -n
                done

                # 2. Construct the base command based on the model name.
                COMMAND="bash train.sh $seed $dataset $model"

                # Check if the model is 'timeMCL' and modify the command accordingly.
                if [ "$model" == "timeMCL" ]; then
                    COMMAND="bash train.sh $seed $dataset $model 16 relaxed-wta"
                fi
                if [ "$model" == "timePrism" ]; then
                    COMMAND="bash train.sh $seed $dataset $model 625 Short"
                fi
                if [ "$model" == "timePrism_iTran" ]; then
                    COMMAND="bash train.sh $seed $dataset $model 64 Short"
                fi
                # 3. Define a unique log file for each job.
                LOG_FILE="${LOG_DIR}/seed-${seed}_data-${dataset}_model-${model}.log"

                # 4. Start the training job in the background.
                echo "Starting: $COMMAND"
                echo "Log -> $LOG_FILE"
                
                # The '&' symbol runs the command in the background.
                # '>' redirects standard output (stdout) to the log file.
                # '2>&1' redirects standard error (stderr) to the same place as stdout.
                $COMMAND > "$LOG_FILE" 2>&1 &
                
                # A short pause to prevent a sudden burst of jobs that might lag the system.
                sleep 1

            done
        done
    done
fi

# --- Cleanup ---
echo "-------------------------------------"
echo "All training jobs have been launched."
echo "Waiting for the remaining background jobs to complete..."

# The 'wait' command without arguments waits for all background jobs started by this script to complete.
wait

echo "--- All training jobs have completed ---"
echo "Log files are saved in the directory: $LOG_DIR"
echo "You can now check the log files to see the results of each job."
