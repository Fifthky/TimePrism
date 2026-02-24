import os
import sys
from pathlib import Path
import warnings
import argparse
import json

# --- Project Path Setup (mirrors plot_similar_windows.py) ---
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

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from omegaconf import OmegaConf, DictConfig
import hydra

from tsExperiments.models.project_models.TimePrism.timePrism_estimator import TimePrismEstimator

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# --- Global Matplotlib Font and Style Settings ---
# Increase all font sizes (titles, axis labels, tick labels, legend, etc.)
plt.rcParams.update(
    {
        "font.size": 14,          # base font size
        "axes.titlesize": 18,     # title font size
        "axes.labelsize": 16,     # x and y labels
        "xtick.labelsize": 14,    # x tick labels
        "ytick.labelsize": 14,    # y tick labels
        "legend.fontsize": 14,    # legend
    }
)


DATASET_NAME_MAP = {
    "exchange": "exchange_rate_nips",
    "electricity": "electricity_nips",
    "traffic": "traffic_nips",
    "solar": "solar_nips",
    "wiki": "wiki-rolling_nips",
    "taxi": "taxi_30min",
}


def find_ckpt_path(cfg: DictConfig, ckpt_data: dict) -> str:
    """
    Find the checkpoint path for TimePrism with the expected 'Short' history.
    """
    history_mode = "Short"
    key = f"seed_{cfg.seed}_{cfg.dataset_name}_{cfg.model_name}_{cfg.num_hypotheses}_hist_{history_mode}"

    ckpt_path = ckpt_data.get(key)
    if not ckpt_path:
        raise FileNotFoundError(
            f"Could not find a valid checkpoint for key pattern matching '{key}' in ckpts.json"
        )

    print(f"Found checkpoint for key '{key}'")
    return ckpt_path


def load_timeprism_and_dataset(cfg: DictConfig):
    """
    Load the TimePrism predictor and the grouped test dataset entry,
    mirroring the logic used in training.
    """
    print("--- Loading Dataset ---")
    full_dataset_name = DATASET_NAME_MAP.get(cfg.dataset_name, cfg.dataset_name)
    dataset_path = Path(cfg.paths.dataset_path)
    dataset = get_dataset(full_dataset_name, path=dataset_path)

    target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
    print(f"Dataset '{full_dataset_name}' loaded. Target dim: {target_dim}")

    grouper = MultivariateGrouper(max_target_dim=target_dim)
    full_dataset_test_list = list(grouper(dataset.test))
    if not full_dataset_test_list:
        raise ValueError("Dataset test split is empty after grouping.")

    full_target_entry = full_dataset_test_list[0]

    print("--- Loading TimePrism Checkpoint ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(cfg.ckpt_json_path, "r") as f:
        ckpt_data = json.load(f)

    ckpt_path_str = find_ckpt_path(cfg, ckpt_data)
    ckpt_path = Path(ckpt_path_str)
    run_dir = ckpt_path.parent.parent

    # Load the original Hydra configuration from the training run.
    hydra_config_dir = run_dir / ".hydra"
    if not hydra_config_dir.is_dir():
        raise FileNotFoundError(f"Could not find .hydra config directory in {run_dir}")

    with hydra.initialize_config_dir(config_dir=str(hydra_config_dir), job_name="plot_prob_job"):
        original_cfg = hydra.compose(config_name="config")
    print("Successfully loaded the original, complete configuration from the training run.")

    # Instantiate the TimePrism estimator mirroring train.py
    estimator_class = TimePrismEstimator
    model_params = dict(original_cfg.model.params)

    estimator = estimator_class(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        context_length=dataset.metadata.prediction_length,
        target_dim=target_dim,
        trainer_kwargs=original_cfg.trainer,
        data_kwargs=original_cfg.data,
        **model_params,
    )

    transformation = estimator.create_transformation()
    training_network = estimator.create_lightning_module()
    predictor_net = training_network.__class__.load_from_checkpoint(str(ckpt_path), map_location=device)
    predictor = estimator.create_predictor(transformation, predictor_net)

    return full_target_entry, predictor, dataset.metadata, target_dim


def get_timeprism_prediction(predictor, data_entry):
    """
    Run TimePrism in evaluation mode and return hypotheses and probabilities.
    """
    device = predictor.prediction_net.device
    transformation = predictor.input_transform

    data_t_generator = transformation.apply([data_entry], is_train=False)
    data_t = next(iter(data_t_generator))

    with torch.no_grad():
        kwargs = {
            "past_target_cdf": torch.tensor(data_t["past_target_cdf"], dtype=torch.float32).unsqueeze(0).to(device),
            "past_observed_values": torch.tensor(data_t["past_observed_values"], dtype=torch.float32).unsqueeze(0).to(
                device
            ),
            "past_time_feat": torch.tensor(data_t["past_time_feat"], dtype=torch.float32).unsqueeze(0).to(device),
            "target_dimension_indicator": torch.tensor(
                data_t["target_dimension_indicator"], dtype=torch.long
            ).unsqueeze(0).to(device),
        }
        hypotheses, probabilities = predictor.prediction_net.forward_for_evaluation(**kwargs)

    return hypotheses.cpu().numpy(), probabilities.cpu().numpy()


def compute_pit_and_coverage(full_target_entry, predictor, metadata, dim_to_plot: int = 0):
    """
    Slide a prediction window over the test series, collect PIT values,
    and compute empirical coverage vs nominal coverage.
    """
    context_len = metadata.prediction_length
    pred_len = metadata.prediction_length

    full_series = full_target_entry["target"][dim_to_plot, :]
    series_len = full_series.shape[-1]

    max_start = series_len - (context_len + pred_len)
    if max_start <= 0:
        raise ValueError("Time series is too short for the chosen context and prediction lengths.")

    pits = []

    for start_idx in tqdm(range(max_start + 1), desc="Computing PIT values"):
        history_end_idx = start_idx + context_len
        future_end_idx = history_end_idx + pred_len

        window_entry = {
            "start": full_target_entry["start"] + start_idx,
            "target": full_target_entry["target"][:, :history_end_idx],
        }

        forecasts, probabilities = get_timeprism_prediction(predictor, window_entry)

        # forecasts: [num_hypotheses, pred_len, target_dim]
        # probabilities: typically 2D for TimePrism; we aggregate to a single
        # weight per hypothesis and reuse that across time steps.
        num_hyp, pred_steps, _ = forecasts.shape

        # Derive per-hypothesis weights from the returned probability tensor.
        probs = np.asarray(probabilities)
        if probs.ndim == 1:
            hyp_weights = probs.astype(float)
        elif probs.ndim == 2:
            hyp_weights = probs.mean(axis=1).astype(float)
        elif probs.ndim == 3:
            hyp_weights = probs.mean(axis=(1, 2)).astype(float)
        else:
            raise ValueError(f"Unexpected probability tensor shape: {probs.shape}")

        # Ground truth future values
        y_future = full_series[history_end_idx:future_end_idx]

        for t in range(pred_steps):
            values = forecasts[:, t, dim_to_plot].astype(float)
            weights = hyp_weights.astype(float)

            # Normalize weights to sum to 1 for safety.
            w_sum = weights.sum()
            if w_sum <= 0:
                continue
            weights = weights / w_sum

            order = np.argsort(values)
            vals_sorted = values[order]
            w_sorted = weights[order]

            y_obs = float(y_future[t])
            mask = vals_sorted <= y_obs
            pit = float(w_sorted[mask].sum())
            pits.append(pit)

    pits = np.array(pits, dtype=float)
    if pits.size == 0:
        raise ValueError("No PIT values were computed; check data and model outputs.")

    # Compute empirical coverage for central intervals using PIT values.
    coverage_levels = np.linspace(0.1, 0.9, 9)  # 0.1, 0.2, ..., 0.9
    empirical_coverage = []

    for c in coverage_levels:
        alpha = 1.0 - c
        lower = alpha / 2.0
        upper = 1.0 - alpha / 2.0
        covered = np.logical_and(pits >= lower, pits <= upper)
        empirical_coverage.append(covered.mean())

    empirical_coverage = np.array(empirical_coverage, dtype=float)

    return pits, coverage_levels, empirical_coverage


def plot_coverage_vs_nominal(coverage_levels, empirical_coverage, dataset_name: str, dim_to_plot: int):
    # Use the same width but half the height for both figures
    plt.figure(figsize=(6, 3))
    plt.plot(coverage_levels, empirical_coverage, "o-", label="Empirical coverage")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title(f"Coverage vs Nominal - {dataset_name}, dim {dim_to_plot}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    # Adjust layout but keep overall figure size fixed
    plt.tight_layout()


def plot_pit_histogram(pits, dataset_name: str, dim_to_plot: int):
    # Match the figure size and aspect ratio to the coverage vs nominal plot
    plt.figure(figsize=(6, 3))
    plt.hist(pits, bins=20, range=(0.0, 1.0), density=True, alpha=0.7, edgecolor="black")
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Uniform density")
    plt.xlabel("PIT value")
    plt.ylabel("Density")
    plt.title(f"PIT Histogram - {dataset_name}, dim {dim_to_plot}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    # Adjust layout but keep overall figure size fixed
    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Coverage vs Nominal and PIT Histogram for TimePrism.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="exchange",
        help="Short name of the dataset (e.g., 'exchange', 'electricity', 'solar').",
    )
    parser.add_argument(
        "--dim_to_plot",
        type=int,
        default=0,
        help="Target dimension (channel) to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3141,
        help="Random seed used for training the TimePrism model.",
    )
    args = parser.parse_args()

    # Fixed configuration for TimePrism 625 Short
    cfg = OmegaConf.create(
        {
            "model_name": "timePrism",
            "dataset_name": args.dataset,
            "dim_to_plot": args.dim_to_plot,
            "num_hypotheses": 625,
            "seed": args.seed,
            "paths": {
                "dataset_path": str(project_root / "tsExperiments" / "gluonts_cache" / "datasets"),
            },
            "ckpt_json_path": str(project_root / "tsExperiments" / "ckpts.json"),
        }
    )

    try:
        full_target_entry, predictor, metadata, target_dim = load_timeprism_and_dataset(cfg)
        pits, coverage_levels, empirical_coverage = compute_pit_and_coverage(
            full_target_entry, predictor, metadata, dim_to_plot=cfg.dim_to_plot
        )

        # Create plots
        plot_coverage_vs_nominal(coverage_levels, empirical_coverage, cfg.dataset_name, cfg.dim_to_plot)
        plot_pit_histogram(pits, cfg.dataset_name, cfg.dim_to_plot)

        # Save figures
        output_dir = Path("./probability_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build dataset name for file naming: first letter uppercase, rest as is (e.g., 'solar' -> 'Solar')
        dataset_title = cfg.dataset_name[0].upper() + cfg.dataset_name[1:]

        # Naming convention:
        #   PIT_<Dataset>_dim_<k>.png
        #   Reliability_<Dataset>_dim_<k>.png
        cov_path = output_dir / f"Reliability_{dataset_title}_dim_{cfg.dim_to_plot}.png"
        pit_path = output_dir / f"PIT_{dataset_title}_dim_{cfg.dim_to_plot}.png"

        plt.figure(1)
        plt.savefig(cov_path, dpi=150)
        plt.figure(2)
        plt.savefig(pit_path, dpi=150)

        print(f"Saved coverage vs nominal plot to: {cov_path}")
        print(f"Saved PIT histogram to: {pit_path}")

    except (FileNotFoundError, ValueError, KeyError, AttributeError) as e:
        print("\n--- An error occurred ---")
        print(e)
        sys.exit(1)


