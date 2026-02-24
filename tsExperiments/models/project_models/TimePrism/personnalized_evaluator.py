import numpy as np
import pandas as pd
from typing import Dict, Iterable, Iterator, List, Tuple, Union, Mapping, Callable, Optional, Any
import os # Import the 'os' module to manage environment variables

from gluonts.model.forecast import Forecast, Quantile
from gluonts.gluonts_tqdm import tqdm
from itertools import chain, tee

# =================================================================================================
# SECTION 1: CORE HELPER FUNCTIONS
# =================================================================================================

def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> np.ndarray:
    """Calculates the pinball loss between target and forecast at a given quantile."""
    err = target - forecast
    return np.maximum(q * err, (q - 1) * err)

# =================================================================================================
# SECTION 2: THE EVALUATOR CLASS
# =================================================================================================

class MultivariateEvaluator:
    """
    A comprehensive evaluator for multivariate forecasts.
    
    This version implements a robust evaluation strategy:
    1. Per-Channel Normalization: Both true targets and forecasts are normalized
       on a per-channel basis using the statistics of the true targets.
    2. Calculation in Normalized Space: All metrics are computed on this normalized data.
    """
    def __init__(
        self,
        quantiles: Iterable[Union[float, str]] = (np.arange(20) / 20.0)[1:],
        model_name: str = "",
        full_test_dataset: Optional[Iterable[Dict[str, Any]]] = None, # No longer used for NED
        prediction_length: Optional[int] = None,
        **kwargs
    ):
        self.quantiles = tuple(map(Quantile.parse, quantiles))
        self.model_name = model_name
        self.prediction_length = prediction_length

    def __call__(
        self,
        ts_iterator: Iterable[Union[pd.DataFrame, pd.Series]],
        fcst_iterator: Iterable[Forecast],
        num_series: Optional[int] = None,
    ) -> Dict[str, float]:
        """Main evaluation entry point."""
        print("--- Reading evaluation data (rolling windows) ---")

        first_fcst, fcst_iterator = self._peek_iterator(fcst_iterator)
        
        if isinstance(first_fcst, dict):
            # This is the Pair path, receiving raw dictionaries
            print("Processing raw Pair data inside evaluator.")
            # We need to re-peek the ts_iterator as well
            _, ts_iterator = self._peek_iterator(ts_iterator)
            targets_list, forecasts_list = self._process_pair_input(
                ts_iterator, fcst_iterator, num_series
            )
        else:
            # This is the standard path for other models
            print("Processing GluonTS Forecast objects inside evaluator.")
            # We need to re-peek the ts_iterator as well
            _, ts_iterator = self._peek_iterator(ts_iterator)
            targets_list, forecasts_list = self._gluonts_to_numpy(
                ts_iterator, fcst_iterator, num_series
            )
        
        agg_metrics = self._compute_metrics(targets_list, forecasts_list)
        
        return agg_metrics
        
    def _process_pair_input(
        self,
        raw_ts_iterator: Iterable[dict],
        forecast_dict_iterator: Iterable[dict],
        num_series: int
    ) -> Tuple[List[np.ndarray], List[dict]]:
        """
        Processes the raw dictionary outputs from the pair evaluation loop.
        This centralizes the data slicing and formatting logic within the evaluator.
        """
        targets_list = []
        # The forecast list is already in the correct format, so we just convert the iterator to a list
        forecasts_list = list(forecast_dict_iterator) 

        for test_entry in tqdm(raw_ts_iterator, total=num_series, desc="Processing pair data entries"):
            # The target is a numpy array of shape (Dims, Time)
            full_target_array = test_entry['target'].T  # Transpose to (Time, Dims)

            # Extract the future part (prediction target)
            pred_target = full_target_array[-self.prediction_length:]
            targets_list.append(pred_target)
            
        return targets_list, forecasts_list
        
    def _gluonts_to_numpy(self, targets_it: Iterable[pd.DataFrame], forecasts_it: Iterable[Forecast], num_series: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Converts GluonTS objects to numpy arrays."""
        targets_list, forecasts_list = [], []
        
        first_forecast, forecasts_iterable = self._peek_iterator(forecasts_it)
        if first_forecast is None: return [], []
            
        for target_df, forecast in tqdm(zip(targets_it, forecasts_iterable), total=num_series, desc="Reading and processing time series data"):
            pred_target = self._extract_pred_target(target_df, forecast)
            targets_list.append(pred_target)
            
            samples = forecast.samples
            if samples.shape[-1] == pred_target.shape[-1] + 1:
                samples = samples[..., :-1]
            forecasts_list.append(samples)
            
        return targets_list, forecasts_list

    def _compute_metrics(self, targets_list, forecasts_list) -> Dict[str, float]:
        """
        Orchestrates evaluation using vectorized calculations for most metrics.
        """
        if not targets_list: 
            print("--- WARNING: No targets available for metric computation. Returning empty metrics. ---")
            return {}
        global_targets_np = np.stack(targets_list, axis=0)      # (B, T, D)
        num_channels = global_targets_np.shape[-1]
        batch_size = global_targets_np.shape[0]

        is_pair = isinstance(forecasts_list[0], dict) and "hypotheses" in forecasts_list[0]

        if is_pair:
            print("--- Detected Pair input format (hypotheses and probabilities) ---")
            global_forecasts_np = np.stack([f["hypotheses"] for f in forecasts_list], axis=0)  # (B, K, T, D)
            weights = np.stack([f["probabilities"] for f in forecasts_list], axis=0)           # (B, K, D)
            
            prob_sums = np.sum(weights[0, :, 0])
            print(f"Verifying probabilities sum for a sample: {prob_sums:.4f}")
            if not np.isclose(prob_sums, 1.0, atol=1e-5):
                print(f"WARNING: Probabilities do not sum to 1! Sum is {prob_sums}. Metrics might be skewed.")
        else:
            print("--- Detected standard sample-based input format ---")
            global_forecasts_np = np.stack(forecasts_list, axis=0)  # (B, K, T, D)
            num_samples = global_forecasts_np.shape[1]
            # Assign equal weights, shape (B, K, D) for consistency
            weights = np.full((batch_size, num_samples, num_channels), 1.0 / num_samples)

        # --- Vectorized Normalization (Identical for both cases) ---
        print("--- Calculating metrics via vectorized operations ---")
        channel_means = np.nanmean(global_targets_np, axis=(0, 1), keepdims=True)
        channel_stds = np.nanstd(global_targets_np, axis=(0, 1), keepdims=True)
        channel_stds[channel_stds < 1e-8] = 1.0
        targets_norm = (global_targets_np - channel_means) / channel_stds
        forecasts_norm = (global_forecasts_np - channel_means) / channel_stds
        abs_target_norm_per_channel = np.nansum(np.abs(targets_norm), axis=(0, 1))

        # --- Unified & Weighted Metrics Calculation ---
        
        # MSE (uses weighted mean for Pair)
        if is_pair:
            # Weighted mean: Sum(P_i * X_i)
            weights_expanded_mse = weights[:, :, np.newaxis, :] # -> (B, K, 1, D)
            mean_forecast_norm = np.sum(forecasts_norm * weights_expanded_mse, axis=1) # -> (B, T, D)
        else:
            mean_forecast_norm = np.nanmean(forecasts_norm, axis=1)
        mse_per_channel = np.nanmean(np.square(targets_norm - mean_forecast_norm), axis=(0, 1))
        print("--- MSE Calculated ---")

        # NMAE (uses weighted median for Pair)
        if is_pair:
            median_forecast_norm = self._weighted_quantile(forecasts_norm, 0.5, weights) # -> (B, T, D)
        else:
            median_forecast_norm = np.nanquantile(forecasts_norm, 0.5, axis=1)
        abs_err_norm_per_channel = np.nansum(np.abs(targets_norm - median_forecast_norm), axis=(0, 1))
        nmae_per_channel = np.divide(abs_err_norm_per_channel, abs_target_norm_per_channel, out=np.zeros_like(abs_err_norm_per_channel, dtype=float), where=abs_target_norm_per_channel != 0)
        print("--- NMAE Calculated ---")

        # CRPS (unified calculation using weighted_crps)
        crps_per_channel = self.weighted_crps(targets_norm, forecasts_norm, weights)
        print("--- CRPS Calculated ---")

        # --- Distortion (Recalculated to match the provided formula) ---
        gt_expanded_norm = np.expand_dims(targets_norm, axis=1)
        
        # Step 1: Calculate Squared Errors. Shape: (B, K, T, D)
        squared_errors = (forecasts_norm - gt_expanded_norm) ** 2
        
        # Step 2: Average over BOTH the Time (axis=2) and Dimension (axis=3) axes.
        # This computes the Mean Squared Error across the entire T*D grid for each hypothesis.
        # Use nanmean here to ignore missing values (NaNs) coming from masked/padded targets.
        mse_per_sample = np.nanmean(squared_errors, axis=(2, 3))
        # Shape: (B, K) – may still contain NaNs if an entire (T,D) slice is missing
        
        # Step 3: Take the square root to get a holistic, per-channel-averaged RMSE for each hypothesis.
        rmse_per_sample = np.sqrt(mse_per_sample)
        # Shape: (B, K)
        # Step 5: Find the minimum RMSE across the K hypotheses for each batch item.
        # This corresponds to the min_k part of the formula.
        distortion_per_batch = np.min(rmse_per_sample, axis=1)
        # Shape: (B,)
        
        print("--- Distortion Calculated ---")

        # --- Final Aggregation ---
        final_metrics = {
            'MSE': np.nanmean(mse_per_channel),
            'NMAE': np.nanmean(nmae_per_channel),
            'CRPS': np.nanmean(crps_per_channel),
            'Distortion': np.nanmean(distortion_per_batch),
        }
        return final_metrics

    @staticmethod
    def _peek_iterator(iterable: Iterable[Any]) -> Tuple[Optional[Any], Iterable[Any]]:
        iterator = iter(iterable)
        try:
            peeked_object = next(iterator)
        except StopIteration:
            return None, iter([]) 
        return peeked_object, chain([peeked_object], iterator)

    @staticmethod
    def _extract_pred_target(time_series: pd.DataFrame, forecast: Forecast) -> np.ndarray:
        return np.atleast_1d(np.squeeze(time_series.loc[forecast.index].values))

    @staticmethod
    def _weighted_quantile(forecasts: np.ndarray, q: float, weights: np.ndarray) -> np.ndarray:
        """
        Calculates the weighted quantile (e.g., median) for a set of forecasts.

        Args:
            forecasts (np.ndarray): Shape (B, K, T, D)
            q (float): The quantile to compute (e.g., 0.5 for median).
            weights (np.ndarray): Shape (B, K, D)

        Returns:
            np.ndarray: The weighted quantile. Shape (B, T, D)
        """
        # Ensure weights are correctly shaped for broadcasting
        weights_expanded = weights[:, :, np.newaxis, :]  # -> (B, K, 1, D)
        
        # Sort forecasts along the sample dimension (K)
        sorted_indices = np.argsort(forecasts, axis=1)
        sorted_forecasts = np.take_along_axis(forecasts, sorted_indices, axis=1)
        
        # Sort weights to match the sorted forecasts
        sorted_weights = np.take_along_axis(weights_expanded, sorted_indices, axis=1)
        
        # Calculate cumulative sum of weights
        cumsum_weights = np.cumsum(sorted_weights, axis=1)
        
        # Find the index where the cumulative sum exceeds the quantile
        # This gives us the index of the quantile value for each (B, T, D) point
        quantile_indices = np.argmax(cumsum_weights >= q, axis=1)
        
        # Gather the quantile values from the sorted forecasts
        # We need to expand quantile_indices to match the dimensions of sorted_forecasts
        quantile_values = np.take_along_axis(sorted_forecasts, quantile_indices[:, np.newaxis, :, :], axis=1).squeeze(1)
        
        return quantile_values

    @staticmethod
    def weighted_crps(
        targets: np.ndarray, 
        forecasts: np.ndarray, 
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the weighted Continuous Ranked Probability Score (CRPS) per channel.

        This implementation uses the energy score formulation of CRPS. The second
        term is calculated using a memory-efficient identity that avoids creating
        the KxK pairwise difference matrix, instead relying on sorting.

        Args:
            targets (np.ndarray): Shape: (B, T, D)
            forecasts (np.ndarray): Shape: (B, K, T, D)
            probabilities (np.ndarray): Shape: (B, K, D)

        Returns:
            np.ndarray: The CRPS score per channel. Shape: (D,)
        """
        # --- Reshape for Broadcasting ---
        targets_expanded = np.expand_dims(targets, axis=1)       # -> (B, 1, T, D)
        probs_expanded = probabilities[:, :, np.newaxis, :]     # -> (B, K, 1, D)

        # --- First term: E[|X - y|] (weighted) ---
        # This part is memory-efficient and remains unchanged.
        abs_diff_term1 = np.abs(forecasts - targets_expanded)
        term1 = np.sum(probs_expanded * abs_diff_term1, axis=1)  # -> (B, T, D)

        # --- OPTIMIZED Second term: 0.5 * E[|X - X'|] (weighted) ---
        # This new implementation avoids the massive (B, K, K, T, D) array.
        
        # Step 1: Sort forecasts and their corresponding probabilities along the sample (K) axis.
        sorted_indices = np.argsort(forecasts, axis=1)
        sorted_forecasts = np.take_along_axis(forecasts, sorted_indices, axis=1)
        
        # Probabilities need to be expanded to be sorted along the same axes as forecasts
        probs_expanded_for_sort = np.broadcast_to(probs_expanded, forecasts.shape)
        sorted_probs = np.take_along_axis(probs_expanded_for_sort, sorted_indices, axis=1)

        # Step 2: Calculate the weighted empirical CDF.
        # cumsum() on sorted probabilities gives the CDF value at each sorted forecast point.
        cdf = np.cumsum(sorted_probs, axis=1) # -> (B, K, T, D)

        # Step 3: Calculate the two main components of the identity.
        # Component A: F(x) * (1 - F(x))
        # We only need the first K-1 terms for the summation.
        cdf_term = cdf[:, :-1, :, :] * (1.0 - cdf[:, :-1, :, :])

        # Component B: (x_(i+1) - x_(i))
        # np.diff computes the difference between adjacent elements.
        forecast_diffs = np.diff(sorted_forecasts, axis=1) # -> (B, K-1, T, D)
        
        # Step 4: Compute the sum for the identity and multiply by 2.
        # This is the expectation E[|X - X'|]
        integrand = cdf_term * forecast_diffs
        expectation_term = 2.0 * np.sum(integrand, axis=1) # Sum over the K-1 dimension -> (B, T, D)

        term2 = 0.5 * expectation_term

        # --- Combine terms to get CRPS grid and average per channel ---
        crps_grid = term1 - term2  # -> (B, T, D)
        return np.mean(crps_grid, axis=(0, 1)) # -> (D,)