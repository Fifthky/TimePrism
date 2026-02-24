# tsExperiments/scripts_plot/plot_similar_windows.py

import os
import sys
from pathlib import Path
import warnings
import argparse
import json

# --- Project Path Setup (Unaltered as requested) ---
try:
    script_dir = Path(__file__).parent.resolve()
    tsExperiments_dir = script_dir.parent.resolve()
    project_root = tsExperiments_dir.parent.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(tsExperiments_dir))
    print(f"Project root added to path: {project_root}")
    print(f"tsExperiments directory added to path: {tsExperiments_dir}")
except NameError:
    project_root = Path.cwd().parent.parent
    tsExperiments_dir = Path.cwd().parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(tsExperiments_dir))

# --- Imports (Unaltered as requested) ---
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from tsExperiments.models.project_models.TimePrism.timePrism_estimator import TimePrismEstimator
from tsExperiments.models.project_models.deepAR.estimator import deepVAREstimator
from tsExperiments.models.project_models.tMCL.timeMCL_estimator import timeMCL_estimator
from tsExperiments.models.project_models.timeGrad.timeGradEstimator import TimEstimatorGrad
from tsExperiments.models.project_models.tempflow.tempFlow_estimator import TempFlowEstimator
from tsExperiments.models.project_models.transformerTempFlow.transformerTempFlow_estimator import TransformerTempFlowEstimator
from tsExperiments.models.project_models.tactis2.estimator import TACTiSEstimator
import hydra
from omegaconf import OmegaConf, DictConfig
# --- Settings and Global Variables (Unaltered as requested) ---
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
DATASET_NAME_MAP = {
    "exchange": "exchange_rate_nips", "electricity": "electricity_nips",
    "traffic": "traffic_nips", "solar": "solar_nips",
    "wiki": "wiki-rolling_nips", "taxi": "taxi_30min"
}
ESTIMATOR_MAP = {
    "timePrism": TimePrismEstimator, "deepAR": deepVAREstimator,
    "timeMCL": timeMCL_estimator, "timeGrad": TimEstimatorGrad,
    "tempflow": TempFlowEstimator, "transformer_tempflow": TransformerTempFlowEstimator,
    "tactis2": TACTiSEstimator,
}

# === HELPER FUNCTIONS ===

def find_ckpt_path(cfg: DictConfig, ckpt_data: dict) -> str:
    """Finds the checkpoint path from the ckpts.json data."""
    # --- MODIFICATION START: Corrected logic for finding checkpoint key ---
    # Default to Full history for most models
    history_mode = "Full"
    # Set specific overrides
    if cfg.model_name == "timePrism":
        history_mode = "Short"

    key = f"seed_{cfg.seed}_{cfg.dataset_name}_{cfg.model_name}_{cfg.num_hypotheses}_hist_{history_mode}"
    
    ckpt_path = ckpt_data.get(key)
    # A fallback for timeMCL which has a more complex name structure
    if not ckpt_path and cfg.model_name == "timeMCL":
        key_mcl = f"seed_{cfg.seed}_{cfg.dataset_name}_{cfg.model_name}_{cfg.num_hypotheses}_relaxed-wta_epsilon_0.1_scaler_mean_hist_{history_mode}"
        ckpt_path = ckpt_data.get(key_mcl)
        if ckpt_path:
            print(f"Found checkpoint for key '{key_mcl}'")
            return ckpt_path
                 
    if not ckpt_path:
        raise FileNotFoundError(f"Could not find a valid checkpoint for key pattern matching '{key}' in ckpts.json")
    
    print(f"Found checkpoint for key '{key}'")
    return ckpt_path
    # --- MODIFICATION END ---


def load_model_and_dataset(cfg: DictConfig):
    """Loads dataset and predictor using the original Hydra config from the training run."""
    # --- MODIFICATION START: Final and correct instantiation logic for ALL models ---
    print("--- Loading Dataset ---")
    full_dataset_name = DATASET_NAME_MAP.get(cfg.dataset_name, cfg.dataset_name)
    dataset_path = Path(cfg.paths.dataset_path)
    dataset = get_dataset(full_dataset_name, path=dataset_path)
    
    target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
    print(f"Dataset '{full_dataset_name}' loaded. Target dim: {target_dim}")
    
    grouper = MultivariateGrouper(max_target_dim=target_dim)
    full_dataset_train_list = list(grouper(dataset.train))
    if not full_dataset_train_list:
        raise ValueError("Dataset train split is empty after grouping.")

    print(f"--- Loading Model Checkpoint: {cfg.model_name} ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(cfg.ckpt_json_path, 'r') as f:
        ckpt_data = json.load(f)

    # --- Special Handling for tactis2 ---
    if cfg.model_name == "tactis2":
        # Step 1: Find the two separate checkpoint paths for tactis2
        base_key = f"seed_{cfg.seed}_{cfg.dataset_name}_{cfg.model_name}_{cfg.num_hypotheses}_hist_Full"
        key_phase1 = f"{base_key}_phase_1"
        key_phase2 = f"{base_key}_phase_2"
        
        ckpt_path_phase1 = ckpt_data.get(key_phase1)
        ckpt_path_phase2 = ckpt_data.get(key_phase2)

        if not ckpt_path_phase1 or not ckpt_path_phase2:
            raise FileNotFoundError(f"Could not find phase 1 or 2 checkpoints for tactis2 with base key '{base_key}'")
        
        print(f"Found Phase 1 checkpoint: {ckpt_path_phase1}")
        print(f"Found Phase 2 checkpoint: {ckpt_path_phase2}")
        
        # Load config from the run directory of the first phase
        run_dir = Path(ckpt_path_phase1).parent.parent
    else:
        # --- Standard logic for all other models ---
        ckpt_path_str = find_ckpt_path(cfg, ckpt_data)
        ckpt_path = Path(ckpt_path_str)
        run_dir = ckpt_path.parent.parent

    # Step 2: Load the complete, original Hydra configuration from the run directory.
    hydra_config_dir = run_dir / ".hydra"
    if not hydra_config_dir.is_dir():
        raise FileNotFoundError(f"Could not find .hydra config directory in {run_dir}")
    
    with hydra.initialize_config_dir(config_dir=str(hydra_config_dir), job_name="plot_job"):
        original_cfg = hydra.compose(config_name="config")
    print("Successfully loaded the original, complete configuration from the training run.")

    # Step 3: Instantiate the estimator by precisely mimicking the logic in train.py
    estimator_class = ESTIMATOR_MAP[cfg.model_name]

    # FIX: Mirror train.py logic regarding use_full_history.
    # Only TimePrism-style models accept this parameter; remove it for others.
    model_params = dict(original_cfg.model.params)
    if cfg.model_name not in ["timePrism", "timePrism_iTran"]:
        model_params.pop("use_full_history", None)

    # Provide ALL required top-level arguments, exactly as in train.py,
    # before unpacking the rest of the parameters from the loaded config.
    estimator = estimator_class(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        context_length=dataset.metadata.prediction_length,
        target_dim=target_dim,
        trainer_kwargs=original_cfg.trainer,
        data_kwargs=original_cfg.data,
        **model_params,
    )
    
    # Step 4: Load the network and build the final predictor
    transformation = estimator.create_transformation()
    training_network = estimator.create_lightning_module()

    # Special loading for tactis2, standard for others
    if cfg.model_name == "tactis2":
        predictor_net = training_network.__class__.load_from_checkpoint(ckpt_path_phase1, map_location=device)
        predictor_net.switch_to_stage_2(predictor_net.model, "adam")
        predictor_net.load_state_dict(torch.load(ckpt_path_phase2, map_location=device)["state_dict"])
    else:
        predictor_net = training_network.__class__.load_from_checkpoint(str(ckpt_path), map_location=device)
        
    predictor = estimator.create_predictor(transformation, predictor_net)
    
    return full_dataset_train_list[0], predictor, dataset.metadata, target_dim
    # --- MODIFICATION END ---

def get_prediction(predictor, data_entry, model_name):
    """Unified prediction interface for TimePrism and other models."""
    
    # --- MODIFICATION START: Added special handling for timeMCL ---
    if model_name == "timeMCL":
        # This logic mimics the special evaluation block for timeMCL in train.py
        
        # Safely get the forecast_generator
        if not hasattr(predictor, "forecast_generator") or not hasattr(predictor.forecast_generator, "sample_hyps"):
            raise AttributeError("The provided predictor for timeMCL does not have a 'sample_hyps' flag.")
            
        forecast_generator = predictor.forecast_generator
        original_sample_hyps_state = forecast_generator.sample_hyps
        
        try:
            # Step 1: Temporarily disable random sampling to get all deterministic hypotheses
            forecast_generator.sample_hyps = False
            print(f"INFO: Temporarily disabled sampling for {model_name} to retrieve all hypotheses.")

            # Step 2: Run prediction. The model will now return its discrete scenarios.
            # The num_samples parameter is typically ignored when sample_hyps is False.
            forecast_it = predictor.predict([data_entry])
            forecast = next(forecast_it)

            # Step 3: timeMCL's hypotheses include a score in the last channel. We must remove it.
            all_hyps_and_scores = forecast.samples
            hypotheses = all_hyps_and_scores[..., :-1]  # Exclude the final score channel

            return hypotheses, None # Return the clean hypotheses

        finally:
            # Step 4: CRITICAL - Always restore the predictor's original state
            forecast_generator.sample_hyps = original_sample_hyps_state
    # --- MODIFICATION END ---
            
    elif model_name == "timePrism":
        # Use the dedicated evaluation forward pass
        device = predictor.prediction_net.device
        transformation = predictor.input_transform
        
        data_t_generator = transformation.apply([data_entry], is_train=False)
        data_t = next(iter(data_t_generator))

        with torch.no_grad():
            # Construct kwargs dict for the forward pass
            kwargs = {
                "past_target_cdf": torch.tensor(data_t['past_target_cdf'], dtype=torch.float32).unsqueeze(0).to(device),
                "past_observed_values": torch.tensor(data_t['past_observed_values'], dtype=torch.float32).unsqueeze(0).to(device),
                "past_time_feat": torch.tensor(data_t['past_time_feat'], dtype=torch.float32).unsqueeze(0).to(device),
                "target_dimension_indicator": torch.tensor(data_t['target_dimension_indicator'], dtype=torch.long).unsqueeze(0).to(device)
            }
            hypotheses, probabilities = predictor.prediction_net.forward_for_evaluation(**kwargs)
            return hypotheses.cpu().numpy(), probabilities.cpu().numpy()
    else:
        # Use standard GluonTS prediction for all other sampling-based models
        forecast_it = predictor.predict([data_entry])
        forecast = next(forecast_it)
        return forecast.samples, None # Return samples and None for probabilities


def find_similar_windows(
    full_series: np.ndarray, 
    query_idx: int, 
    context_len: int,
    pred_len: int,
    num_windows: int = 5,
    min_separation: int = 20 # New parameter: minimum steps between selected windows
):
    """
    Finds N historical windows that are similar to a query window but are
    temporally well-separated.

    This function uses a greedy iterative approach to ensure separation.
    """
    query_window = full_series[query_idx : query_idx + context_len]
    
    # Calculate the distance of ALL possible windows to the query window once.
    all_distances = []
    search_space = len(full_series) - context_len - pred_len
    for i in range(search_space):
        candidate_window = full_series[i : i + context_len]
        distance = np.linalg.norm(query_window - candidate_window)
        all_distances.append({'index': i, 'distance': distance})
        
    # Sort all windows by their distance to the query window (most similar first)
    all_distances.sort(key=lambda x: x['distance'])
    
    # --- MODIFICATION START: Iterative selection with exclusion zones ---
    selected_indices = []
    # Convert the sorted list to a DataFrame for easier filtering
    candidates_df = pd.DataFrame(all_distances)
    
    print("\n--- Iterative Window Selection ---")
    while len(selected_indices) < num_windows and not candidates_df.empty:
        # Step 1: Greedily select the current best candidate (most similar)
        best_candidate = candidates_df.iloc[0]
        best_index = int(best_candidate['index'])
        selected_indices.append(best_index)
        
        print(f"  - Selected window at index {best_index} (distance: {best_candidate['distance']:.4f})")
        
        # Step 2: Create an "exclusion zone" around the selected index.
        # Any candidate starting within this zone will be removed.
        exclusion_start = best_index - min_separation
        exclusion_end = best_index + min_separation
        
        # Step 3: Remove all candidates that fall within the exclusion zone.
        # We also remove the selected candidate itself.
        candidates_df = candidates_df[
            (candidates_df['index'] < exclusion_start) | 
            (candidates_df['index'] > exclusion_end)
        ]
    
    print(f"Final selected window indices: {selected_indices}")
    # --- MODIFICATION END ---
    
    return selected_indices


# === MAIN PLOTTING FUNCTION ===

def plot_similar_windows(
    full_target_entry, predictor, metadata,
    model_name: str = "timePrism",
    dim_to_plot: int = 0,
    target_dim: int = 0, # Added target_dim
    query_idx: int = -200, # A recent point in history
):
    """Main function to find similar windows and plot them with forecasts."""
    context_len = metadata.prediction_length
    pred_len = metadata.prediction_length
    
    full_series = full_target_entry['target'][dim_to_plot, :]
    
    print(f"Finding 5 windows most similar to the window at index {query_idx}...")
    similar_indices = find_similar_windows(
        full_series, query_idx, context_len, pred_len=pred_len, num_windows=5
    )
    
    # Sort the selected indices chronologically for plotting
    similar_indices.sort()
    print(f"Plotting windows in chronological order: {similar_indices}")
    
    # --- MODIFICATION START: Adjust plotting setup for 5 plots and new style ---
    # Setup for 1 row, 5 columns, with a much shorter (flatter) aspect ratio
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True) # Greatly reduced height
    
    # Create a single, complete DatetimeIndex for the entire series
    full_series_index = pd.date_range(
        start=full_target_entry['start'].to_timestamp(), 
        periods=len(full_series), 
        freq=metadata.freq
    )
    
    for i, start_idx in enumerate(tqdm(similar_indices, desc="Plotting similar windows")):
        ax = axes[i]
        
        # --- Prepare data for this window ---
        history_end_idx = start_idx + context_len
        future_end_idx = history_end_idx + pred_len
        
        window_entry = {
            "start": full_target_entry['start'] + start_idx,
            "target": full_target_entry['target'][:, :history_end_idx]
        }
        
        # --- Get Prediction ---
        forecasts, probabilities = get_prediction(predictor, window_entry, model_name)

        # --- Plotting ---
        context_index = full_series_index[start_idx:history_end_idx]
        future_index = full_series_index[history_end_idx:future_end_idx]
        
        # --- Add colored background panes ---
        ax.axvspan(context_index[0], context_index[-1], facecolor='#D9D9D9', alpha=0.5, zorder=-10)
        ax.axvspan(future_index[0], future_index[-1], facecolor='#E2F0D9', alpha=0.5, zorder=-10)
        
        # --- Plot continuous history and ground truth ---
        # Combine the last point of history with the future ground truth for a continuous line
        continuous_gt_index = full_series_index[history_end_idx-1:future_end_idx]
        continuous_gt_values = full_series[history_end_idx-1:future_end_idx]

        ax.plot(context_index, full_series[start_idx:history_end_idx], color='black', linewidth=2.0)
        ax.plot(continuous_gt_index, continuous_gt_values, color='black', linewidth=2.0) # Plot GT connecting to history

        # Plot forecasts based on model type
        if model_name == "timePrism":
            top_k = 10
            top_k = min(top_k, forecasts.shape[0])
            avg_probs = np.mean(probabilities, axis=1)
            
            top_indices = np.argsort(avg_probs)[-top_k:]
            
            top_hypotheses = forecasts[top_indices, :, dim_to_plot]
            top_probabilities = avg_probs[top_indices]

            min_prob, max_prob = top_probabilities.min(), top_probabilities.max()
            if (max_prob - min_prob) > 1e-9:
                 normalized_probs = (top_probabilities - min_prob) / (max_prob - min_prob)
            else:
                 normalized_probs = np.ones(top_k)

            min_linewidth = 1.5 # Slightly thicker lines
            max_linewidth = 4.0
            
            cmap = mcolors.LinearSegmentedColormap.from_list("prob_cmap", ["#D76975", "#517BC8"])
            
            for j in range(top_k):
                color = cmap(normalized_probs[j])
                linewidth = min_linewidth + normalized_probs[j] * (max_linewidth - min_linewidth)
                
                # Prepend the last historical point to each forecast for a continuous line
                forecast_values = np.concatenate(([full_series[history_end_idx-1]], top_hypotheses[j, :]))
                ax.plot(continuous_gt_index, forecast_values, color=color, alpha=0.7, linewidth=linewidth)
        else: # For other models
            # --- MODIFICATION START: Added mean forecast plot for other models ---
            num_samples_to_plot = min(100, forecasts.shape[0])
            
            # Plot all individual samples
            for j in range(num_samples_to_plot):
                forecast_values = np.concatenate(([full_series[history_end_idx-1]], forecasts[j, :, dim_to_plot]))
                if model_name == "timeMCL":
                    ax.plot(continuous_gt_index, forecast_values, color='#1D4999', alpha=0.2, linewidth=1.5)
                else:
                    ax.plot(continuous_gt_index, forecast_values, color='#B4C7E7', alpha=0.2)

            # Calculate and plot the mean of the samples
            mean_forecast = np.mean(forecasts[:, :, dim_to_plot], axis=0)
            mean_forecast_values = np.concatenate(([full_series[history_end_idx-1]], mean_forecast))
            ax.plot(continuous_gt_index, mean_forecast_values, color='#2F5597', linestyle='--', linewidth=3.0)
            
            # --- MODIFICATION END ---

        # --- Apply final styling ---
        # Remove titles
        ax.set_title("")
        
        # Increase font sizes and add thicker grid
        ax.tick_params(axis='x', rotation=30, labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(True, which='both', linestyle='-', linewidth='1.2', color='white')
        labels = ax.get_xticklabels()
        # We iterate through the labels and set the visibility.
        for j, label in enumerate(labels):
            if j in [1,3,5,7,8]:
                label.set_visible(False) # Hide the label
        
    # Remove the main title
    # fig.suptitle("") # No need for this line, just don't create one
    
    plt.subplots_adjust(wspace=0.05, bottom=0.3) 
    
    # Save the figure
    output_dir = Path("./future_scenarios/")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"similar_windows_{cfg.dataset_name}_{cfg.model_name}_dim{cfg.dim_to_plot}.png"
    
    plt.savefig(save_path, dpi=150) # Use a good DPI for clarity
    print(f"Plot saved to {save_path}")
    plt.show()
    # --- MODIFICATION END ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and plot forecasts for similar historical windows.")
    parser.add_argument("--model", type=str, required=True, help="Short name of the model (e.g., 'timePrism').")
    parser.add_argument("--dataset", type=str, default="exchange", help="Short name of the dataset (e.g., 'exchange').")
    parser.add_argument("--dim_to_plot", type=int, default=0, help="The dimension (channel) to plot.")
    parser.add_argument("--num_hypotheses", type=int, default=1, help="Number of hypotheses the model was trained with.")
    parser.add_argument("--seed", type=int, default=3141, help="The random seed used for training.")
    
    args = parser.parse_args()

    # Create a DictConfig object for internal consistency with the functions
    cfg = OmegaConf.create({
        "model_name": args.model,
        "dataset_name": args.dataset,
        "dim_to_plot": args.dim_to_plot,
        "num_hypotheses": args.num_hypotheses,
        "seed": args.seed,
        "paths": {
            "dataset_path": str(project_root / "tsExperiments" / "gluonts_cache" / "datasets")
        },
        "ckpt_json_path": str(project_root / "tsExperiments" / "ckpts.json")
    })
    
    try:
        full_target_entry, predictor, metadata, target_dim = load_model_and_dataset(cfg)
        plot_similar_windows(
            full_target_entry, predictor, metadata,
            model_name=cfg.model_name,
            dim_to_plot=cfg.dim_to_plot,
            target_dim=target_dim
        )
    except (FileNotFoundError, ValueError, KeyError, AttributeError) as e:
        print(f"\n--- An error occurred ---")
        print(e)
        sys.exit(1)