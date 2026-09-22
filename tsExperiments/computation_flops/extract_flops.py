import mlflow
import pandas as pd
import os
import argparse
import sys
from pathlib import Path


def extract_flops_summary(mlflow_tracking_uri, experiment_name, output_csv_path):
    """
    Searches an MLflow experiment for FLOPs calculation runs, extracts the
    results, and generates a summary CSV file.
    """
    print(f"Setting MLflow tracking URI to: '{mlflow_tracking_uri}'")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    print(f"Searching for runs in MLflow experiment: '{experiment_name}'")
    try:
        runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time"])
    except mlflow.exceptions.RestException as e:
        print(f"Error: Could not find or access MLflow experiment '{experiment_name}'.")
        print(f"Details: {e}")
        sys.exit(1)

    if runs.empty:
        print("No runs found in the specified experiment. Exiting.")
        return

    # --- Data Extraction and Cleaning ---
    
    flops_runs = runs.dropna(subset=['metrics.prediction_flops']).copy()
    
    if flops_runs.empty:
        print("No runs with the 'prediction_flops' metric were found. Exiting.")
        return

    # --- FIX: Use a more robust regex to handle complex run names ---
    # This new regex non-greedily matches the content between the model name
    # and the '_compute_ps_' anchor, making it robust to variations.
    run_name_regex = r"seed_\d+_[a-zA-Z0-9-]+_([a-zA-Z0-9_]+)_\d+_.+?_compute_ps_(\d+)"
    
    extracted_data = flops_runs['tags.mlflow.runName'].str.extract(run_name_regex)
    
    flops_runs['model_name'] = extracted_data[0]
    flops_runs['num_parallel_samples'] = pd.to_numeric(extracted_data[1])
    
    flops_runs.dropna(subset=['model_name', 'num_parallel_samples'], inplace=True)

    if flops_runs.empty:
        print("Could not parse model name or sample count from any run names. Check run name format.")
        return

    print(f"Found {len(flops_runs)} valid FLOPs calculation runs to process.")

    # --- Create the Pivot Table ---
    
    summary_table = flops_runs.pivot_table(
        index='model_name',
        columns='num_parallel_samples',
        values='metrics.prediction_flops',
        aggfunc='first' 
    )
    
    # NOTE: We now use the raw `summary_table` for FLOPs, without division.
    
    summary_table = summary_table.sort_index(axis=1)

    # --- Save to CSV ---
    
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- MODIFICATION START: Reformat the DataFrame before saving ---

    # Step 1: Define the desired public names and column order.
    public_name_map = {
       "transformer_tempflow": "Trf.Flow", "tactis2": "Tactis2",
        "timeGrad": "TimeGrad", "deepAR": "DeepAR", "tempflow": "TempFlow",
        "timeMCL": "TimeMCL", "timePrism": "TimePrism",
    }
    
    # This list defines the final order of columns (using internal names).
    final_column_order = [
        "deepAR", "timeGrad", "tempflow", "transformer_tempflow",
        "tactis2", "timeMCL", "timePrism"
    ]

    # Step 2: Transpose the table. Rows are now samples, columns are internal model names.
    final_table = summary_table.T
    
    # Step 3: Rename the columns from internal names to public display names.
    final_table = final_table.rename(columns=public_name_map)
    
    # Step 4: Reorder the columns according to the desired final order.
    final_column_order_public = [public_name_map.get(name, name) for name in final_column_order]
    columns_to_keep = [col for col in final_column_order_public if col in final_table.columns]
    final_table = final_table[columns_to_keep]

    # Step 5: Define a helper function to format numbers into LaTeX scientific notation.
    def format_to_latex_sci_notation(x):
        """Formats a number to 2-significant-figure scientific notation for LaTeX."""
        if not (pd.notna(x) and isinstance(x, (int, float))):
            return ''  # Return an empty string for non-numeric or NaN values
        if x == 0:
            return "$0$" # Handle the zero case
        
        # Format to scientific notation with one decimal place (total 2 significant figures)
        sci_str = f"{x:.1e}"
        
        # Split into mantissa and exponent
        mantissa, exponent = sci_str.split('e')
        
        # Construct the final LaTeX string, wrapped in $$
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"

    # Step 6: Apply the formatting to all numeric cells in the DataFrame.
    formatted_table = final_table.applymap(format_to_latex_sci_notation)
    
    # Step 7: Convert the index (which is float by default) to integer type.
    formatted_table.index = formatted_table.index.astype(int)
    
    # Step 8: Rename the index column to 'Sampling Times'.
    formatted_table.index.name = "Sampling Times"
    
    # Step 9: Save the completely reformatted DataFrame to CSV.
    formatted_table.to_csv(output_csv_path)
    
    print("\n--- FLOPs Summary Table (Raw FLOPs, LaTeX formatted) ---")
    print(formatted_table.to_string())
    print(f"\nSuccessfully saved formatted summary to: {output_csv_path}")
    # --- MODIFICATION END ---


if __name__ == "__main__":
    # --- Setup argument parser ---
    parser = argparse.ArgumentParser(description="Extract FLOPs summary from MLflow experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="compute_exchange_1",
        help="Name of the MLflow experiment to analyze. (Default: compute_exchange_1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flops_summary.csv",  # Default path relative to this script
        help="Path to save the output CSV file. (Default: flops_summary.csv)",
    )
    args = parser.parse_args()

    # --- Automatically determine MLflow URI based on file structure ---
    # This script is in '.../computation_flops/'
    # The mlruns directory is in '../logs/mlflow/mlruns/'
    # We construct the absolute path to be safe.
    script_dir = Path(__file__).parent.resolve()
    mlruns_path = script_dir.parent / "logs" / "mlflow" / "mlruns"

    if not mlruns_path.exists():
        print("Error: Could not find the 'mlruns' directory at the expected path:")
        print(f"  {mlruns_path}")
        print("Please ensure your project structure is correct.")
        sys.exit(1)

    # Convert the path to a URI format that MLflow understands
    mlflow_tracking_uri = mlruns_path.as_uri()

    # Construct the absolute output path
    absolute_output_path = (script_dir / args.output).resolve()

    extract_flops_summary(mlflow_tracking_uri, args.experiment, str(absolute_output_path))
