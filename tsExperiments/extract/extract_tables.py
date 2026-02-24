# %% Imports and Setup
import os
import pandas as pd
import numpy as np
import sys
import warnings
import rootutils
import argparse
from pathlib import Path
import math
# %% Display and warning settings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
np.seterr(all="ignore")

# Add the project root to the path
try:
    root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except rootutils.RootutilsError:
    root = Path.cwd().parent 
    os.environ["PROJECT_ROOT"] = str(root)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))


# %% Helper Functions
def format_significant(mean, std):
    """
    Format the mean to 3 significant figures and the std to 2 significant
    figures, preserving trailing zeros.
    """
    
    def format_single_number(num, significant_figures):
        """
        Helper to format one number to a specified number of significant figures.
        
        Args:
            num (float): The number to format.
            significant_figures (int): The number of significant figures to keep.
        """
        if num == 0:
            # Create a string with a zero and the correct number of decimal places
            return "0." + "0" * (significant_figures - 1)

        # Calculate the position of the first significant digit
        power = math.floor(math.log10(abs(num)))
        
        # The number of decimal places needed is (significant_figures - 1) minus this power.
        # Example (num=12.34, sig_figs=3): power=1, decimals=(3-1)-1=1 -> "12.3"
        # Example (num=0.1234, sig_figs=3): power=-1, decimals=(3-1)-(-1)=3 -> "0.123"
        decimal_places = (significant_figures - 1) - power
        
        # Ensure we don't have negative decimal places for large numbers
        if decimal_places < 0:
            decimal_places = 0
            
        # Use the 'f' formatter, which respects the specified number of decimal places
        return f"{num:.{decimal_places}f}"

    # Call the helper function with 3 for the mean
    mean_formatted = format_single_number(mean, 3)
    # Call the helper function with 2 for the standard deviation
    std_formatted = f"{std:.2f}"
    
    return f"{mean_formatted} $\\pm$ {std_formatted}"

def get_seeds_from_args(args):
    """Derive the list of seeds from parsed CLI arguments."""
    if len(args.seeds) == 1 and args.seeds[0].lower() == "all":
        print("Processing with all default seeds: [3141, 3142, 3143]")
        return [3141, 3142, 3143]
    try:
        seeds_list = [int(seed) for seed in args.seeds]
        print(f"Processing with specified seeds: {seeds_list}")
        return seeds_list
    except ValueError:
        raise ValueError("Invalid seed provided. All seeds must be integer numbers.")

def look_exact_path(path_pattern):
    """Find the exact file path in the directory matching the given path pattern."""
    root_dir = os.path.dirname(path_pattern)
    base_filename = os.path.basename(path_pattern).split('.csv')[0]
    if not os.path.exists(root_dir): return None
    for file in os.listdir(root_dir):
        if file.startswith(base_filename): return os.path.join(root_dir, file)
    return None

# %% Main Extraction Logic
def extract_and_process_results(seeds):
    """Main function to extract, process, and save the results."""
    
    # --- Configuration ---
    datasets_list = ["electricity", "exchange", "solar", "traffic", "wiki"]
    
    model_configs = {
        "ETS": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "deepAR": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "timeGrad": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "tempflow": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "transformer_tempflow": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "tactis2": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "timeMCL": {"num_hypotheses": 16, "suffix": "hist_Full"},
        # TimePrism-16 (N=16, Short history) extracted separately from standard TimePrism (N=625).
        "timePrism_16": {
            "base_model_key": "timePrism",
            "num_hypotheses": 16,
            "suffix": "hist_Short",
        },
        "timePrism": {
            "base_model_key": "timePrism",
            "num_hypotheses": 625,
            "suffix": "hist_Short",
        },
    }

    public_name_map = {
        "ETS": "ETS", "transformer_tempflow": "Trf.Flow", "tactis2": "Tactis2",
        "timeGrad": "TimeGrad", "deepAR": "DeepAR", "tempflow": "TempFlow",
        "timeMCL": "TimeMCL",
        "timePrism_16": "TimePrism-16",
        "timePrism": "TimePrism",
    }
    dataset_name_map = {
        "electricity": "Elec.", "exchange": "Exch.", "solar": "Sol.",
        "traffic": "Traf.", "wiki": "Wiki.",
    }
    
    metrics_to_extract = {
        "CRPS": "CRPS", "Distortion": "Distortion",
        "NMAE": "NMAE", "MSE": "MSE",
    }
    
    results = {
        metric: pd.DataFrame(
            index=[dataset_name_map[d] for d in datasets_list],
            columns=[public_name_map[m] for m in model_configs.keys()],
        ) for metric in metrics_to_extract.keys()
    }
    
    # --- Data Extraction Loop ---
    print(f"Processing results for seeds: {seeds}")
    
    for dataset_name in datasets_list:
        path_pattern = f"{os.environ['PROJECT_ROOT']}/tsExperiments/results/saved_csv/eval_{dataset_name}_200.csv"
        exact_path = look_exact_path(path_pattern)
        
        if not exact_path:
            print(f"Warning: CSV file not found for dataset '{dataset_name}'. Skipping.")
            continue
        
        try:
            full_csv = pd.read_csv(exact_path, low_memory=False)
            full_csv['_start_time'] = pd.to_datetime(full_csv['_start_time'], errors='coerce')
            print(f"Successfully loaded data from: {exact_path}")
        except Exception as e:
            print(f"Error reading {exact_path}: {e}")
            continue

        for model_key, params in model_configs.items():
            suffix = params["suffix"]
            base_model_key = params.get("base_model_key", model_key)
            valid_runs_for_aggregation = []
            
            for seed in seeds:
                # Filter by model name first
                model_df = full_csv[full_csv['Name'].str.contains(base_model_key, na=False, case=False)]
                if base_model_key == "tempflow":
                    model_df = model_df[~model_df['Name'].str.contains("transformer_tempflow", na=False, case=False)]
                # --- End of FIX ---
                if base_model_key == "timeMCL":
                    # Apply special filter for timeMCL if necessary
                    model_df = model_df[model_df['Name'].str.contains('relaxed', na=False)]
                
                # Filter by seed
                seed_df = model_df[model_df['Name'].str.contains(f'seed_{seed}_', na=False)]

                # Define criteria for a valid run
                num_hyp = params["num_hypotheses"]
                hyp_check_str = f"_{num_hyp}_"
                
                # Apply all filters to find all valid runs for the current seed
                valid_runs_for_seed = seed_df[
                    seed_df['Name'].str.contains(hyp_check_str, na=False) &
                    seed_df['Name'].str.endswith(suffix)
                ]

                # If any valid runs are found, get the latest one among them
                if not valid_runs_for_seed.empty:
                    latest_run = valid_runs_for_seed.sort_values(by='_start_time', ascending=False).iloc[0]
                    valid_runs_for_aggregation.append(latest_run)

            if not valid_runs_for_aggregation:
                continue
            
            final_runs_df = pd.DataFrame(valid_runs_for_aggregation)
            
            public_col_name = public_name_map[model_key]
            public_row_name = dataset_name_map[dataset_name]

            for metric_alias, metric_col in metrics_to_extract.items():
                # Default to an empty string. It will only be overwritten if valid data is found.
                final_result = "" 
                if metric_col in final_runs_df.columns:
                    metric_values = pd.to_numeric(final_runs_df[metric_col], errors='coerce')
                    valid_points = metric_values.dropna()
                    
                    if len(valid_points) == 1:
                        mean_val = valid_points.iloc[0]
                        std_val = 0.0
                        final_result = format_significant(mean_val, std_val)
                    elif len(valid_points) > 1:
                        mean_val = valid_points.mean()
                        std_val = valid_points.std()
                        final_result = format_significant(mean_val, std_val)
                    # If len(valid_points) is 0, final_result remains an empty string, NOT "N/A".
                
                results[metric_alias].loc[public_row_name, public_col_name] = final_result

    # --- Save to CSV ---
    output_dir = Path("./experiment_results_csv")
    output_dir.mkdir(exist_ok=True)
    
    for metric_alias, df in results.items():
        output_path = output_dir / f"results_{metric_alias}.csv"
        # Fill any potential NaN values from DataFrame creation with empty strings before saving
        df.fillna('', inplace=True)
        df.T.to_csv(output_path)
        print(f"Successfully saved results to {output_path}")


def extract_and_process_results_fev(seeds):
    """Extract, process, and save the results for FEV datasets."""
    
    # --- Configuration for FEV datasets ---
    
    model_configs = {
        "ETS": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "deepAR": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "timeGrad": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "tempflow": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "transformer_tempflow": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "tactis2": {"num_hypotheses": 1, "suffix": "hist_Full"},
        "timeMCL": {"num_hypotheses": 16, "suffix": "hist_Full"},
        "timePrism": {"num_hypotheses": 625, "suffix": "hist_Short"},
    }

    public_name_map = {
        "ETS": "ETS", "transformer_tempflow": "Trf.Flow", "tactis2": "Tactis2",
        "timeGrad": "TimeGrad", "deepAR": "DeepAR", "tempflow": "TempFlow",
        "timeMCL": "TimeMCL", "timePrism": "TimePrism",
    }
    datasets_list = ["uci_air", "hospital", "mdense", "hierachi"]
    dataset_name_map = {
        "uci_air": "UCI", "hospital": "Hosp.", "mdense": "M-Den.", "hierachi": "Hier.",
    }
    
    metrics_to_extract = {
        "CRPS": "CRPS", "Distortion": "Distortion",
        "NMAE": "NMAE", "MSE": "MSE",
    }
    
    results = {
        metric: pd.DataFrame(
            index=[dataset_name_map[d] for d in datasets_list],
            columns=[public_name_map[m] for m in model_configs.keys()],
        ) for metric in metrics_to_extract.keys()
    }
    
    # --- Data Extraction Loop ---
    print(f"Processing FEV results for seeds: {seeds}")
    
    for dataset_name in datasets_list:
        path_pattern = f"{os.environ['PROJECT_ROOT']}/tsExperiments/results/saved_csv/eval_{dataset_name}_200.csv"
        exact_path = look_exact_path(path_pattern)
        
        if not exact_path:
            print(f"Warning: CSV file not found for FEV dataset '{dataset_name}'. Skipping.")
            continue
        
        try:
            full_csv = pd.read_csv(exact_path, low_memory=False)
            full_csv['_start_time'] = pd.to_datetime(full_csv['_start_time'], errors='coerce')
            print(f"Successfully loaded FEV data from: {exact_path}")
        except Exception as e:
            print(f"Error reading {exact_path}: {e}")
            continue

        for model_key, params in model_configs.items():
            suffix = params["suffix"] 
            valid_runs_for_aggregation = []
            
            for seed in seeds:
                # Filter by model name first
                model_df = full_csv[full_csv['Name'].str.contains(model_key, na=False, case=False)]
                if model_key == "tempflow":
                    model_df = model_df[~model_df['Name'].str.contains("transformer_tempflow", na=False, case=False)]
                if model_key == "timeMCL":
                    model_df = model_df[model_df['Name'].str.contains('relaxed', na=False)]
                
                # Filter by seed
                seed_df = model_df[model_df['Name'].str.contains(f'seed_{seed}_', na=False)]

                # Define criteria for a valid run
                num_hyp = params["num_hypotheses"]
                hyp_check_str = f"_{num_hyp}_"
                
                valid_runs_for_seed = seed_df[
                    seed_df['Name'].str.contains(hyp_check_str, na=False) &
                    seed_df['Name'].str.endswith(suffix)
                ]

                if not valid_runs_for_seed.empty:
                    latest_run = valid_runs_for_seed.sort_values(by='_start_time', ascending=False).iloc[0]
                    valid_runs_for_aggregation.append(latest_run)

            if not valid_runs_for_aggregation:
                continue
            
            final_runs_df = pd.DataFrame(valid_runs_for_aggregation)
            
            public_col_name = public_name_map[model_key]
            public_row_name = dataset_name_map[dataset_name]

            for metric_alias, metric_col in metrics_to_extract.items():
                final_result = "" 
                if metric_col in final_runs_df.columns:
                    metric_values = pd.to_numeric(final_runs_df[metric_col], errors='coerce')
                    valid_points = metric_values.dropna()
                    
                    if len(valid_points) == 1:
                        mean_val = valid_points.iloc[0]
                        std_val = 0.0
                        final_result = format_significant(mean_val, std_val)
                    elif len(valid_points) > 1:
                        mean_val = valid_points.mean()
                        std_val = valid_points.std()
                        final_result = format_significant(mean_val, std_val)
                
                results[metric_alias].loc[public_row_name, public_col_name] = final_result
    
    # --- Save to CSV ---
    output_dir = Path("./experiment_results_csv_fev")
    output_dir.mkdir(exist_ok=True)
    
    for metric_alias, df in results.items():
        output_path = output_dir / f"results_{metric_alias}.csv"
        df.fillna('', inplace=True)
        df.T.to_csv(output_path)
        print(f"Successfully saved FEV results to {output_path}")


def extract_timeprism_n_table(seeds=[3141, 3142, 3143]):
    """
    Extract CRPS and Distortion for TimePrism across N for selected datasets,
    and combine with FLOPs ratios into a compact table.
    """
    print(f"Extracting TimePrism N-sweep table with seeds={seeds}")

    # Datasets and display names (order matters for the final table).
    datasets = ["solar", "electricity", "exchange"]
    dataset_display = {
        "solar": "Solar",
        "electricity": "Electricity",
        "exchange": "Exchange",
    }

    # --- Load FLOPs summary for TimePrism ---
    project_root = Path(os.environ["PROJECT_ROOT"])
    flops_path = project_root / "tsExperiments" / "computation_flops" / "flops_summary_timePrism.csv"

    if not flops_path.exists():
        print(f"Error: FLOPs summary file not found at: {flops_path}")
        return

    flops_df = pd.read_csv(flops_path)
    if not {"n", "flops"}.issubset(flops_df.columns):
        print("Error: FLOPs summary file must contain 'n' and 'flops' columns.")
        return

    flops_df = flops_df.dropna(subset=["n", "flops"])
    flops_df["n"] = flops_df["n"].astype(int)
    flops_df["flops"] = flops_df["flops"].astype(float)
    flops_df = flops_df.sort_values("n")

    if (flops_df["n"] == 1).sum() == 0:
        print("Error: FLOPs summary does not contain an entry for N=1 (needed as baseline).")
        return

    base_flops = float(flops_df.loc[flops_df["n"] == 1, "flops"].iloc[0])

    # --- Load evaluation CSVs for each dataset once ---
    eval_data = {}
    for dataset in datasets:
        path_pattern = f"{os.environ['PROJECT_ROOT']}/tsExperiments/results/saved_csv/eval_{dataset}_200.csv"
        exact_path = look_exact_path(path_pattern)

        if not exact_path:
            print(f"Warning: CSV file not found for dataset '{dataset}'. Skipping.")
            continue

        try:
            df = pd.read_csv(exact_path, low_memory=False)
            df["_start_time"] = pd.to_datetime(df["_start_time"], errors="coerce")
            eval_data[dataset] = df
            print(f"Loaded evaluation data for dataset '{dataset}' from: {exact_path}")
        except Exception as e:
            print(f"Error reading {exact_path}: {e}")

    # --- Build table rows ---
    header_row_1 = ["N", "FLOPs"]
    header_row_1 += [dataset_display[d] for d in datasets for _ in (0, 1)]

    header_row_2 = ["", ""]
    header_row_2 += ["CRPS", "Distortion"] * len(datasets)

    table_rows = [header_row_1, header_row_2]

    for _, row in flops_df.iterrows():
        N_val = int(row["n"])
        flops_val = float(row["flops"])

        if N_val == 1:
            flops_ratio_str = "1.0x"
        else:
            ratio = flops_val / base_flops
            flops_ratio_str = f"{ratio:.1f}x"

        row_values = [str(N_val), flops_ratio_str]

        for dataset in datasets:
            df = eval_data.get(dataset)
            crps_values = []
            distortion_values = []

            if df is not None and "Name" in df.columns:
                # Filter to TimePrism runs with given hypothesis count N and hist_Short suffix.
                model_df = df[df["Name"].str.contains("timePrism", na=False, case=False)]
                model_df = model_df[model_df["Name"].str.contains(f"_{N_val}_", na=False)]
                model_df = model_df[model_df["Name"].str.contains("hist_Short", na=False)]

                if not model_df.empty:
                    # Collect data for all seeds
                    for seed in seeds:
                        seed_df = model_df[model_df["Name"].str.contains(f"seed_{seed}_", na=False)]
                        if not seed_df.empty:
                            latest_row = seed_df.sort_values(by="_start_time", ascending=False).iloc[0]
                            
                            if "CRPS" in latest_row:
                                try:
                                    crps_val = pd.to_numeric(latest_row["CRPS"], errors='coerce')
                                    if pd.notna(crps_val):
                                        crps_values.append(float(crps_val))
                                except Exception:
                                    pass
                            if "Distortion" in latest_row:
                                try:
                                    distortion_val = pd.to_numeric(latest_row["Distortion"], errors='coerce')
                                    if pd.notna(distortion_val):
                                        distortion_values.append(float(distortion_val))
                                except Exception:
                                    pass

            # Format using format_significant function
            crps_formatted = ""
            if len(crps_values) == 1:
                mean_val = crps_values[0]
                std_val = 0.0
                crps_formatted = format_significant(mean_val, std_val)
            elif len(crps_values) > 1:
                mean_val = np.mean(crps_values)
                std_val = np.std(crps_values)
                crps_formatted = format_significant(mean_val, std_val)

            distortion_formatted = ""
            if len(distortion_values) == 1:
                mean_val = distortion_values[0]
                std_val = 0.0
                distortion_formatted = format_significant(mean_val, std_val)
            elif len(distortion_values) > 1:
                mean_val = np.mean(distortion_values)
                std_val = np.std(distortion_values)
                distortion_formatted = format_significant(mean_val, std_val)

            row_values.extend([crps_formatted, distortion_formatted])

        table_rows.append(row_values)

    # --- Save the table ---
    output_dir = Path("./experiment_results_csv")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "timePrism_n_table.csv"

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(output_path, index=False, header=False)

    print(f"Successfully saved TimePrism N-sweep table to {output_path}")

# %% Script Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract experiment results and build result tables.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["all"],
        help='Specify one or more seed numbers, or use "all" for default seeds (3141, 3142, 3143).',
    )
    parser.add_argument(
        "--timePrism_n",
        action="store_true",
        help="If set, build the TimePrism N-vs-FLOPs & metrics table instead of the standard tables.",
    )
    args = parser.parse_args()

    if args.timePrism_n:
        # Use the same seed handling as other functions
        active_seeds = get_seeds_from_args(args)
        extract_timeprism_n_table(seeds=active_seeds)
    else:
        active_seeds = get_seeds_from_args(args)
        extract_and_process_results(active_seeds)
        extract_and_process_results_fev([3141])