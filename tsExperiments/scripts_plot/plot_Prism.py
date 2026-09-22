# tsExperiments/scripts_plot/plot_Prism.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# --- Helper function to avoid code duplication if needed elsewhere, otherwise can be kept internal ---
def _prepare_timeseries_dataframe(target_entry, metadata):
    """Converts a raw data entry into a pandas DataFrame with a proper DatetimeIndex."""
    start_period = target_entry['start']
    freq = start_period.freq
    start_timestamp = start_period.to_timestamp()
    
    return pd.DataFrame(
        target_entry['target'].T, 
        index=pd.date_range(start=start_timestamp, periods=target_entry['target'].shape[1], freq=freq)
    )

def plot_prism(
    forecast_dict: dict, 
    target_entry: dict, 
    metadata: object, 
    target_dim: int,
    save_path: Path,
    top_k: int = 16,
    low_prob_color: str = "#C00000",
    high_prob_color: str = "#2F5597"
):
    """
    Generates and saves a specialized plot for TimePrism, showing the top k
    most probable hypotheses with a color gradient.

    Args:
        forecast_dict (dict): A dictionary containing 'hypotheses' and 'probabilities'.
        target_entry (dict): The raw data entry from the dataset, containing 'target' and 'start'.
        metadata (object): The dataset metadata, expected to have a 'prediction_length' attribute.
        target_dim (int): The total number of dimensions in the time series.
        save_path (Path): A Path object pointing to the desired output file (e.g., Path("plot.png")).
        top_k (int, optional): The number of top hypotheses to plot. Defaults to 16.
        low_prob_color (str, optional): Hex color for the lowest probability hypothesis.
        high_prob_color (str, optional): Hex color for the highest probability hypothesis.
    """
    print("--- Generating specialized plots for TimePrism (Top-k Hypotheses) ---")

    # --- Data Extraction and Preparation ---
    hypotheses = forecast_dict['hypotheses']      # Shape: (K, T, D)
    probabilities = forecast_dict['probabilities']  # Shape: (K, D)

    # Average probabilities across dimensions to get a single score per hypothesis
    avg_probabilities = np.mean(probabilities, axis=1) # Shape: (K,)

    # Determine the actual number of top hypotheses to plot
    top_k = min(top_k, len(avg_probabilities))
    if top_k == 0:
        print("Warning: No hypotheses to plot. Aborting plot generation.")
        return
        
    # Get the indices of the top k hypotheses (sorted from lowest to highest probability)
    top_indices = np.argsort(avg_probabilities)[-top_k:]
    top_hypotheses = hypotheses[top_indices, :, :]
    
    # Generate colors based on RANK, not on actual probability values, for a smooth gradient
    colors_for_plot = np.linspace(0, 1, top_k)
    cmap = mcolors.LinearSegmentedColormap.from_list("prob_cmap", [low_prob_color, high_prob_color])

    # --- Plotting Setup ---
    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    # Prepare the full time series DataFrame and slice it into context and ground truth
    full_series_df = _prepare_timeseries_dataframe(target_entry, metadata)
    p_len = metadata.prediction_length
    context_data = full_series_df[-(p_len * 2) : -p_len]
    ground_truth_data = full_series_df[-p_len:]

    # --- Generate Subplots ---
    for i in range(min(rows * cols, target_dim)):
        ax = axes[i]
        
        # Plot historical context data
        ax.plot(context_data.index, context_data.iloc[:, i], color='black', label='History')
        
        # Plot the ground truth for the forecast period
        ax.plot(ground_truth_data.index, ground_truth_data.iloc[:, i], color='black', label='Ground Truth')

        # Plot the top k hypotheses with gradient colors
        forecast_index = ground_truth_data.index
        for j in range(top_k):
            ax.plot(forecast_index, top_hypotheses[j, :, i], color=cmap(colors_for_plot[j]), alpha=0.7)

        ax.set_title(f"Dimension {i}")
        ax.tick_params(axis='x', rotation=45)
        if i == 0:
            ax.legend() # Add a legend only to the first subplot

    # --- Finalize and Save ---
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved TimePrism plot to {save_path}")
    plt.close()