import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import pytorch_lightning as pl
from lightning.pytorch.loggers import Logger
import numpy as np
import torch
import rootutils
from typing import List
import pickle
import copy
from gluonts.dataset.common import ListDataset
import pandas as pd

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

from tsExperiments.models.project_models.tactis2.estimator import TACTiSEstimator
from tsExperiments.models.project_models.timeGrad.timeGradEstimator import (
    TimEstimatorGrad,
)
from tsExperiments.models.project_models.tMCL.timeMCL_estimator import timeMCL_estimator
from tsExperiments.models.project_models.deepAR.estimator import (
    deepVAREstimator
)
from tsExperiments.models.project_models.tempflow.tempFlow_estimator import (
    TempFlowEstimator,
)
from tsExperiments.models.project_models.transformerTempFlow.transformerTempFlow_estimator import (
    TransformerTempFlowEstimator,
)
from tsExperiments.models.project_models.TimePrism.timePrism_estimator import (
    TimePrismEstimator,
)
from tsExperiments.models.project_models.TimePrism_iTran.timePrism_estimator import (
    TimePrismEstimator as TimePrism_iTranEstimator,
)
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from tsExperiments.models.project_models.TimePrism.personnalized_evaluator import (
    MultivariateEvaluator,
)
from utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    split_train_val,
)
from pathlib import Path
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName


log = RankedLogger(__name__, rank_zero_only=True)

def create_zeroed_history_dataset(
    original_dataset: Dataset, context_length: int
) -> ListDataset:
    """
    Creates a new dataset where the history of the target, prior to the
    context_length, is zeroed out.

    Args:
        original_dataset: The original GluonTS dataset.
        context_length: The number of recent time steps to keep untouched.

    Returns:
        A new ListDataset with modified historical data.
    """
    log.info(
        f"Creating new dataset with history before last {context_length} steps zeroed out."
    )
    modified_entries = []
    if not hasattr(original_dataset, '__iter__') or not isinstance(next(iter(original_dataset), None), dict):
        log.warning(f"Dataset is not in the expected format (iterable of dicts). Skipping zeroing.")
        return original_dataset

    for entry in original_dataset:
        # Create a deep copy to avoid modifying the original data in place
        new_entry = copy.deepcopy(entry)
        
        target = new_entry[FieldName.TARGET]
        
        # Calculate the index before which data should be zeroed
        # For a target of length L, we want to keep the last `context_length` points.
        # So we zero out elements from index 0 up to L - context_length.
        zero_until_index = len(target) - context_length
        
        if zero_until_index > 0:
            target[:zero_until_index] = 0
            
        new_entry[FieldName.TARGET] = target
        modified_entries.append(new_entry)
        
    freq = original_dataset.metadata.freq if hasattr(original_dataset, 'metadata') and hasattr(original_dataset.metadata, 'freq') else None
    return ListDataset(modified_entries, freq=freq)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision('high')

    pl.seed_everything(cfg.seed)

    if cfg.data.train.type == "Gluonts_ds":
        try:
            dataset = get_dataset(
                cfg.data.train.dataset_name, regenerate=False, path=Path(cfg.paths.dataset_path)
            )
            metadata = dataset.metadata
        except:
            print(f"Dataset {cfg.data.train.dataset_name} not found, regenerating...")
            dataset = get_dataset(cfg.data.train.dataset_name, regenerate=True)
            metadata = dataset.metadata
        target_dim = min(2000, int(metadata.feat_static_cat[0].cardinality))
        print(f"Data successfully loaded, target_dim: {target_dim}")

        train_grouper = MultivariateGrouper(
            max_target_dim=target_dim
        )
        test_grouper = MultivariateGrouper(
            num_test_dates=int(len(dataset.test) / len(dataset.train)),
            max_target_dim=target_dim,
        )
        log.info(
            f"Using {int(len(dataset.test)/len(dataset.train))} rolling windows (blocks) for testing, as per original logic."
        )

        dataset_train = train_grouper(dataset.train)
        dataset_test = test_grouper(dataset.test)

        if cfg.model.name == "ETS":
            cfg.data.train.split_train_val = False

        if "split_train_val" in cfg.data.train and cfg.data.train.split_train_val:
            log.info("Splitting train and validation datasets")
            dataset_train, dataset_val = split_train_val(
                dataset_name=cfg.data.train.dataset_name,
                grouped_train=dataset_train,
                n_pred_steps_val=cfg.data.train.n_pred_steps_val,
                data_path=cfg.paths.dataset_path
            )

    elif cfg.data.train.type == "fev":
        import pickle
        
        # Determine dataset short name from config
        dataset_name_mapping = {
            'uci_air_quality_1H': 'uci_air',
            'hospital_admissions_1D': 'hospital',
            'M_DENSE_1D': 'mdense',
            'hierarchical_sales_1D': 'hierachi',
        }
        
        short_name = dataset_name_mapping.get(cfg.data.train.dataset_name, cfg.data.train.dataset_name)
        processed_dir = Path(cfg.paths.root_dir) / "tsExperiments" / "fev_cache" / "processed" / short_name
        
        log.info(f"Loading pre-processed fev dataset from: {processed_dir}")
        
        # Load pre-processed data
        with open(processed_dir / "train.pkl", "rb") as f:
            train_list_univariate = pickle.load(f)
        
        with open(processed_dir / "test.pkl", "rb") as f:
            all_test_windows = pickle.load(f)
        
        with open(processed_dir / "metadata.pkl", "rb") as f:
            fev_metadata = pickle.load(f)
        
        num_series = fev_metadata['num_series']
        target_dim = num_series  # Each univariate series becomes one dimension in multivariate
        
        log.info(f"Loaded {num_series} univariate series from pre-processed data")
        log.info(f"FEV data: {num_series} univariate series will be grouped into 1 multivariate series with {target_dim} dimensions")
        
        # Create metadata from saved fev metadata
        from gluonts.dataset.common import MetaData
        metadata = MetaData(
            freq=fev_metadata['freq'],
            prediction_length=fev_metadata['horizon'],
        )
        
        # Convert lists to ListDataset for grouping
        from gluonts.dataset.common import ListDataset
        dataset_train_univariate = ListDataset(train_list_univariate, freq=metadata.freq)
        
        # Flatten all test windows into a single list for proper multivariate grouping
        # Each test window contains the same series but with different cutoff dates
        all_test_entries = []
        for test_window in all_test_windows:
            all_test_entries.extend(test_window)
        
        dataset_test_univariate_flat = ListDataset(all_test_entries, freq=metadata.freq)
        
        log.info(f"Flattened test data: {len(all_test_entries)} total entries from {len(all_test_windows)} windows")
        
        if cfg.model.name == "ETS":
            cfg.data.train.split_train_val = False
        
        # For all models with FEV data, we need to use all series for both train and val
        # to avoid target_dim mismatches (splitting series creates different target_dim for train/val)
        # Split train/val BEFORE multivariate grouping if needed
        if "split_train_val" in cfg.data.train and cfg.data.train.split_train_val:
            # For all models with FEV data: use all series for both train and val to maintain same target_dim
            log.info("For FEV data: using all series for both train and val to maintain consistent target_dim")
            train_split_univariate = train_list_univariate
            val_split_univariate = train_list_univariate
            
            log.info(f"Split univariate data: {len(train_split_univariate)} train, {len(val_split_univariate)} val series")
            
            # Group the univariate data once to create the multivariate dataset
            # Then reuse the same grouped dataset for both train and val to ensure consistent batching
            train_grouper = MultivariateGrouper(max_target_dim=target_dim)
            grouped_dataset = train_grouper(ListDataset(train_split_univariate, freq=metadata.freq))
            
            # Use the same grouped dataset for both train and val to ensure identical structure
            # This prevents batching inconsistencies that can occur when creating separate groupers
            dataset_train = grouped_dataset
            dataset_val = grouped_dataset
            
            test_grouper = MultivariateGrouper(
                num_test_dates=fev_metadata['num_windows'],
                max_target_dim=target_dim,
            )
            dataset_test = test_grouper(dataset_test_univariate_flat)
            
            log.info(f"FEV data grouped to multivariate format, target_dim: {target_dim}")
        else:
            # No split - just group all training data
            train_grouper = MultivariateGrouper(max_target_dim=target_dim)
            test_grouper = MultivariateGrouper(
                num_test_dates=fev_metadata['num_windows'],
                max_target_dim=target_dim,
            )
            
            dataset_train = train_grouper(dataset_train_univariate)
            dataset_test = test_grouper(dataset_test_univariate_flat)
            
            log.info(f"FEV data grouped to multivariate format, target_dim: {target_dim}")

    if cfg.model.name in ["timeGrad"]:
        if cfg.data.train.dataset_name=="wiki-rolling_nips": 
            log.info(f"Overriding batch_size for {cfg.model.name} on {cfg.data.train.dataset_name} dataset to 100.")
            OmegaConf.update(cfg, "data.batch_size", 100, merge=True)
    if cfg.model.name in ["timePrism", "timePrism_iTran"]:
        if cfg.data.train.dataset_name=="wiki-rolling_nips": 
            log.info(f"Overriding batch_size for {cfg.model.name} on {cfg.data.train.dataset_name} dataset to 50.")
            OmegaConf.update(cfg, "data.batch_size", 50, merge=True)
        else:
            log.info(f"Overriding batch_size for {cfg.model.name} on {cfg.data.train.dataset_name} dataset to 100.")
            OmegaConf.update(cfg, "data.batch_size", 100, merge=True) 
    if "+model.params.use_full_history" in sys.argv or "model.params.use_full_history" in cfg.model.params:
         if not cfg.model.params.use_full_history:
            log.info(f"use_full_history is False for model {cfg.model.name}. Zeroing out lagged history for all datasets.")
            
            # Use metadata.prediction_length as it is the base for context_length for all models.
            context_len = metadata.prediction_length

            if 'dataset_train' in locals():
                dataset_train = create_zeroed_history_dataset(
                    original_dataset=dataset_train,
                    context_length=context_len
                )
            
            if "split_train_val" in cfg.data.train and cfg.data.train.split_train_val and 'dataset_val' in locals():
                dataset_val = create_zeroed_history_dataset(
                    original_dataset=dataset_val,
                    context_length=context_len
                )
                
            if 'dataset_test' in locals():
                 dataset_test = create_zeroed_history_dataset(
                    original_dataset=dataset_test,
                    context_length=context_len
                )


    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer_kwargs = {}
    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs["logger"] = logger

    # Ajouter les autres trainer_kwargs
    for key, value in cfg.trainer.items():
        if key not in ["callbacks", "logger"]:
            trainer_kwargs[key] = value

    # instanciate in function of the model called...
    model_name = cfg.model.name
    model_params = cfg.model.params
    # keep use_full_history for both TimePrism variants
    if model_name not in ["timePrism", "timePrism_iTran"]:
        params_copy = {k: v for k, v in model_params.items()}
        params_copy.pop("use_full_history", None)
        model_params = params_copy

    # ------------------------------------------------------------------
    # Special handling of timeMCL + FEV data: align time-feature dims.
    if model_name == "timeMCL" and cfg.data.train.type == "fev":
        from tsExperiments.utils.utils import fourier_time_features_from_frequency

        # metadata.freq is 'D' for hospital, 'H' for uci_air, etc.
        fourier_feats = fourier_time_features_from_frequency(metadata.freq)
        # Each FourierDateFeatures produces a pair (sin, cos), so the
        # actual number of time-feature channels is 2 * len(fourier_feats)
        num_time_feat_channels = 2 * len(fourier_feats)

        old_n_dyn = model_params.get("num_feat_dynamic_real", 0)
        if old_n_dyn != num_time_feat_channels:
            log.info(
                f"[timeMCL+FEV] Overriding num_feat_dynamic_real from "
                f"{old_n_dyn} to {num_time_feat_channels} to match Fourier "
                f"time features for freq='{metadata.freq}'."
            )
            model_params["num_feat_dynamic_real"] = num_time_feat_channels


    if model_name == "timeMCL":
        estimator = timeMCL_estimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "timeGrad":
        estimator = TimEstimatorGrad(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "deepAR":

        log.info(
            f"Setting the output dimension of the distribution to the data dimension: {metadata.feat_static_cat[0].cardinality}"
        )
        model_params["dist_params"]["dim"] = target_dim

        estimator = deepVAREstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "tempflow":

        estimator = TempFlowEstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "transformer_tempflow":


        estimator = TransformerTempFlowEstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "tactis2":
        # Adjust bagging_size if it exceeds target_dim (number of series)
        # This prevents AssertionError in tactis2 model initialization
        if "model_parameters" in model_params and "bagging_size" in model_params["model_parameters"]:
            bagging_size = model_params["model_parameters"]["bagging_size"]
            if bagging_size > target_dim:
                log.info(f"Adjusting bagging_size from {bagging_size} to {target_dim} (target_dim={target_dim})")
                model_params["model_parameters"]["bagging_size"] = target_dim
        
        estimator = TACTiSEstimator(
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            prediction_length=metadata.prediction_length,
            freq=metadata.freq,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )

    elif model_name == "ETS":
        from tsExperiments.models.project_models.ETS.model import ETSForecastModel
        from tsExperiments.models.project_models.ETS.utils import (
            creating_target_list,
            forecast_ets,
        )

        dataset_train = creating_target_list(dataset_train)[0]  # the dataset for training
        estimator = ETSForecastModel(
            forecast_steps=metadata.prediction_length,
            context_length=metadata.prediction_length,
        )
    elif model_name == "timePrism":
        estimator = TimePrismEstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )
    elif model_name == "timePrism_iTran":
        estimator = TimePrism_iTranEstimator(
            freq=metadata.freq,
            prediction_length=metadata.prediction_length,
            target_dim=target_dim,
            context_length=metadata.prediction_length,
            trainer_kwargs=trainer_kwargs,
            data_kwargs=cfg.data,
            **model_params,
        )
        
    else:
        raise ValueError(f"Model {model_name} not supported")

    object_dict = {
        "cfg": cfg,
        "callbacks": callbacks,
        "model": (
            estimator.create_lightning_module() if cfg.model.name != "ETS" else None
        ),
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict, logger)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Mapping checkpoint to device: {device}")
    if cfg.ckpt_path is not None or (
        cfg.ckpt_path_phase1 is not None and cfg.ckpt_path_phase2 is not None
    ):
        transformation = estimator.create_transformation()
        training_network = estimator.create_lightning_module()
        if cfg.model.name == "tactis2":
            assert (
                cfg.train is False
            ), "Retraining Tactis2 from checkpoint is not supported"
            log.info(f"Loading checkpoint from {cfg.ckpt_path_phase1}")
            predictor = training_network.__class__.load_from_checkpoint(
                cfg.ckpt_path_phase1, map_location=device
            )
            log.info(f"Loading checkpoint from {cfg.ckpt_path_phase2}")
            predictor.switch_to_stage_2(predictor.model, "adam")
            predictor.load_state_dict(torch.load(cfg.ckpt_path_phase2)["state_dict"])
        else:
            log.info(f"Loading checkpoint from {cfg.ckpt_path}")
            predictor = training_network.__class__.load_from_checkpoint(cfg.ckpt_path, map_location=device)
        log.info(f"Creating predictor")
        predictor = estimator.create_predictor(transformation, predictor)

    if cfg.train is True:
        log.info(f"Training the model")
        if cfg.model.name == "ETS":
            # Handle NaN values in the dataset before fitting ETS
            # Forward fill then backward fill to handle any missing values
            if dataset_train.isna().any().any():
                log.info(f"Handling NaN values in dataset: forward fill + backward fill")
                dataset_train = dataset_train.ffill().bfill()
            estimator.fit(dataset_train)
        elif "split_train_val" in cfg.data.train :
            predictor = estimator.train(
                training_data=dataset_train,
                validation_data=dataset_val if not cfg.data.get("discard_validation", False) else None,
                ckpt_path=cfg.ckpt_path if cfg.ckpt_path is not None else None,
            )
        else:
            predictor = estimator.train(
                training_data=dataset_train,
                ckpt_path=cfg.ckpt_path if cfg.ckpt_path is not None else None,
            )

    if ((cfg.data.train.type == "Gluonts_ds" and cfg.test is True) or 
        (cfg.data.train.type == "financial_data" and cfg.test is True) or
        (cfg.data.train.type == "fev" and cfg.test is True)):
            log.info(f"Evaluating the model")
                        # --- ADDED: Print total and trainable parameters ---
            if cfg.model.name != "ETS":
                try:
                    total_params = sum(p.numel() for p in predictor.prediction_net.parameters())
                    trainable_params = sum(p.numel() for p in predictor.prediction_net.parameters() if p.requires_grad)
                    log.info(f"--- Model Parameters ({cfg.model.name}) ---")
                    log.info(f"Total parameters: {total_params:,}")
                    log.info(f"Trainable parameters: {trainable_params:,}")
                    log.info(f"------------------------------------")
                except Exception as e:
                    log.warning(f"Could not calculate model parameters: {e}")
            # --- END OF ADDED BLOCK ---
            if cfg.model.name == "ETS":
                targets = creating_target_list(dataset_test)
                forecasts = forecast_ets(
                    target_list=targets,
                    context_length=metadata.prediction_length,
                    trained_model=estimator,
                    num_samples=100, # Using 100 samples for ETS
                    pred_length=metadata.prediction_length,
                )
            elif cfg.model.name == "timePrism":
                # ADDED: Import the new encapsulated function
                from tsExperiments.models.project_models.TimePrism.timePrism_estimator import (
                    run_timeprism_evaluation,
                )

                # Call the clean, encapsulated function
                forecasts, targets = run_timeprism_evaluation(
                    predictor=predictor,
                    estimator=estimator,  # The estimator is available in this scope
                    dataset_test=dataset_test,
                )
            elif cfg.model.name == "timePrism_iTran":
                from tsExperiments.models.project_models.TimePrism_iTran.timePrism_estimator import (
                    run_timeprism_evaluation as run_timeprism_iTran_evaluation,
                )

                forecasts, targets = run_timeprism_iTran_evaluation(
                    predictor=predictor,
                    estimator=estimator,
                    dataset_test=dataset_test,
                )
            else:
                # --- This is the original, unchanged logic for all other models ---
                num_samples_for_prediction = 100
                log.info(f"Generating {num_samples_for_prediction} samples for prediction for model {cfg.model.name}")
                
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=dataset_test,
                    predictor=predictor,
                    num_samples=num_samples_for_prediction,
                )
                forecasts = list(forecast_it)
                targets = list(ts_it)

            ### Computing the metrics
            # The evaluator now needs to know the model name to apply special logic
            evaluator = MultivariateEvaluator(
                quantiles=(np.arange(20) / 20.0)[1:],
                model_name=cfg.model.name,
                full_test_dataset=dataset_test, # Pass the entire test set
                prediction_length=metadata.prediction_length # Pass prediction length for slicing
            )
            
            if cfg.compute_usual_metrics:
                agg_metric = evaluator(
                    targets, forecasts, num_series=len(dataset_test)
                )
            else:
                agg_metric = {}
            # --- ADDED: Recalculate Distortion for timeMCL using all hypotheses ---
            if cfg.compute_usual_metrics and cfg.model.name == "timeMCL":
                log.info("Recalculating Distortion for timeMCL using all hypotheses.")
                
                # Temporarily disable hypothesis sampling to get all of them
                original_sample_hyps = predictor.forecast_generator.sample_hyps
                predictor.forecast_generator.sample_hyps = False

                log.info("Generating all hypotheses for Distortion calculation...")
                forecast_it_all_hyps, ts_it_all_hyps = make_evaluation_predictions(
                    dataset=dataset_test,
                    predictor=predictor,
                    num_samples=cfg.model.params.num_hypotheses,
                )

                # The generator yields SampleForecast objects, where samples are (K, T, D+1)
                # We need to convert this to the dictionary format the evaluator expects for probabilistic forecasts
                forecasts_all_hyps_raw = list(forecast_it_all_hyps)
                
                targets_for_distortion_raw = list(ts_it_all_hyps)
                targets_for_distortion = []
                for ts_df in targets_for_distortion_raw:
                    # _process_pair_input expects a dict with a 'target' key
                    # containing a numpy array of shape (Dims, Time)
                    targets_for_distortion.append({'target': ts_df.values.T})
                
                forecasts_for_distortion = []
                for forecast in forecasts_all_hyps_raw:
                    # forecast.samples has shape (K, T, D+1), where K is num_hypotheses
                    all_hyps_and_scores = forecast.samples
                    hypotheses = all_hyps_and_scores[..., :-1]  # Shape: (K, T, D)
                    
                    # Scores are normalized and repeated over time dimension. Take from the first time step.
                    probabilities = all_hyps_and_scores[:, 0, -1]  # Shape: (K,)
                    
                    # The evaluator expects per-channel probabilities. Since our scores are per-hypothesis, we repeat them for each channel.
                    num_dims = hypotheses.shape[-1]
                    probabilities_expanded = np.repeat(probabilities[:, np.newaxis], num_dims, axis=1) # Shape: (K, D)

                    forecasts_for_distortion.append({
                        "hypotheses": hypotheses,
                        "probabilities": probabilities_expanded
                    })

                # Create a new evaluator instance and compute metrics with all hypotheses
                evaluator_for_distortion = MultivariateEvaluator(
                    quantiles=(np.arange(20) / 20.0)[1:],
                    model_name=cfg.model.name,
                    prediction_length=metadata.prediction_length
                )
                
                agg_metric_distortion_only = evaluator_for_distortion(
                    targets_for_distortion, 
                    forecasts_for_distortion, 
                    num_series=len(dataset_test)
                )
                
                # Update the original metrics dictionary with the new Distortion value
                if 'Distortion' in agg_metric_distortion_only:
                    log.info(f"Original Distortion (from samples): {agg_metric.get('Distortion', 'N/A')}")
                    log.info(f"New Distortion (from all hyps): {agg_metric_distortion_only['Distortion']}")
                    agg_metric['Distortion'] = agg_metric_distortion_only['Distortion']
                else:
                    log.warning("Could not recalculate Distortion for timeMCL.")

                # Restore the original predictor setting
                predictor.forecast_generator.sample_hyps = original_sample_hyps
            for key, value in agg_metric.items():
                for logg in logger:
                    if hasattr(logg, "log_metrics"):
                        logg.log_metrics({key: value})
                    else:
                        log.info(f"Logger {logg} does not have a log_metrics method.")

            if cfg.compute_usual_metrics:
                log.info(f"Final Metrics:")
                log.info(f"  CRPS: {agg_metric.get('CRPS', 'N/A')}")
                log.info(f"  Distortion: {agg_metric.get('Distortion', 'N/A')}")
                log.info(f"  NMAE: {agg_metric.get('NMAE', 'N/A')}")
                log.info(f"  MSE: {agg_metric.get('MSE', 'N/A')}")

    ### Plotting the forecasts
    if cfg.model.plot_forecasts:
        if model_name in ["timePrism", "timePrism_iTran"]:
            # Step 1: Import the newly encapsulated plotting function
            from tsExperiments.scripts_plot.plot_Prism import plot_prism

            # Step 2: Ensure forecast data exists
            if "forecasts" not in locals() or "targets" not in locals():
                log.warning(
                    "Forecasts and targets not found. Running evaluation to generate them for plotting."
                )
                if model_name == "timePrism":
                    from tsExperiments.models.project_models.TimePrism.timePrism_estimator import (
                        run_timeprism_evaluation,
                    )

                    forecasts, targets = run_timeprism_evaluation(
                        predictor=predictor,
                        estimator=estimator,
                        dataset_test=dataset_test,
                    )
                else:
                    from tsExperiments.models.project_models.TimePrism_iTran.timePrism_estimator import (
                        run_timeprism_evaluation as run_timeprism_iTran_evaluation,
                    )

                    forecasts, targets = run_timeprism_iTran_evaluation(
                        predictor=predictor,
                        estimator=estimator,
                        dataset_test=dataset_test,
                    )

            # Step 3: Define the save path
            if model_name == "timePrism":
                plot_save_path = (
                    Path(cfg.paths.output_dir) / "timeprism_top_hypotheses_plot.png"
                )
            else:
                plot_save_path = (
                    Path(cfg.paths.output_dir)
                    / "timeprism_iTran_top_hypotheses_plot.png"
                )

            # Step 4: Call the plotting function with the required data
            plot_prism(
                forecast_dict=forecasts[0],  # The first forecast result
                target_entry=targets[0],  # The first corresponding target entry
                metadata=metadata,  # Dataset metadata
                target_dim=target_dim,  # Total number of dimensions
                save_path=plot_save_path,  # Where to save the plot
                top_k=16,  # Optional: number of hypotheses to show
            )

        # --- This is the original, unchanged plotting logic for all other models ---
        elif model_name != "ETS": # ETS has no forecast plotting implemented here
            if 'forecasts' not in locals() or 'targets' not in locals():
                if cfg.test is False:
                    forecast_it, ts_it = make_evaluation_predictions(
                        dataset=dataset_test,
                        predictor=predictor,
                        num_samples=cfg.model.params.num_hypotheses,
                    )
                    forecasts = list(forecast_it)
                    targets = list(ts_it)

            from tsExperiments.scripts_plot.plottimeMCL import plot_mcl


            target_df = targets[0]
            hypothesis_forecasts = forecasts[
                0
            ].samples  # shape (N, forecast_length, target_dim)

            # Check if we're using timeMCL or not
            is_mcl = cfg.model.name == "timeMCL"
            is_mcl_like = cfg.model.name in ["timeMCL", "timePrism", "timePrism_iTran"]

            # Save all the data to produce the plots

            # Save target_df as pickle instead of CSV and Numpy binary
            with open(f"{cfg.paths.output_dir}/target_df.pkl", "wb") as f:
                pickle.dump(target_df, f)

            # Save other NumPy arrays and metadata as pickle
            with open(f"{cfg.paths.output_dir}/hypothesis_forecasts.pkl", "wb") as f:
                pickle.dump(hypothesis_forecasts, f)

            with open(f"{cfg.paths.output_dir}/forecast_length.pkl", "wb") as f:
                pickle.dump(metadata.prediction_length, f)

            with open(f"{cfg.paths.output_dir}/context_points.pkl", "wb") as f:
                pickle.dump(metadata.prediction_length, f)

            with open(f"{cfg.paths.output_dir}/freq_type.pkl", "wb") as f:
                pickle.dump(metadata.freq, f)

            with open(f"{cfg.paths.output_dir}/is_mcl.pkl", "wb") as f:
                pickle.dump(is_mcl_like, f)

            plot_mcl(
                target_df=target_df,
                hypothesis_forecasts=hypothesis_forecasts,
                forecast_length=metadata.prediction_length,
                context_points=metadata.prediction_length,
                rows=3,
                cols=2,
                plot_mean=True,
                freq_type=metadata.freq,
                save_path=cfg.paths.output_dir,
                is_mcl=is_mcl_like,
                extract_unique=is_mcl_like,
            )
        

    if cfg.visualize_specific_date is True:
        from tsExperiments.scripts_plot.train_viz import plotting_from_a_date,creating_target_list,plot_forecasts_for_dimension
        test_data = dataLoader.creating_test_dataset(start_date=cfg.start_date_viz,end_date=cfg.end_date_viz,num_tests=1)
        targets_loaded = creating_target_list(test_data) 
        if cfg.model.name == "timeMCL":
            contexte_df, forecast_array,start_date,probabilities = plotting_from_a_date(date_of_pred=cfg.date_of_pred,
                                                                        plot_context_size=metadata.prediction_length,
                                                                        target_list=targets_loaded,
                                                                        pred_length=metadata.prediction_length,
                                                                        trained_model=predictor,
                                                                        num_samples=1000,
                                                                        is_mcl = True) 
        else:
             contexte_df, forecast_array,start_date,probabilities = plotting_from_a_date(date_of_pred=cfg.date_of_pred,
                                                                        plot_context_size=metadata.prediction_length,
                                                                        target_list=targets_loaded,
                                                                        pred_length=metadata.prediction_length,
                                                                        trained_model=predictor,
                                                                        num_samples=cfg.model.params.num_hypotheses,
                                                                        is_mcl = False)

        # for dimension_to_plot in [1,2,5,13]:
        for dimension_to_plot in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
            model_name = cfg.model.name
            fig_path = f"{os.environ['PROJECT_ROOT']}/tsExperiments/scripts_plot/{model_name}/{dimension_to_plot}" 
            # fig_path = f"{os.environ['PROJECT_ROOT']}/tsExperiments/logs/plots/{model_name}/{dimension_to_plot}" 
            # mkdir the folder if it doesn't exist
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plot_forecasts_for_dimension(contexte_df, forecast_array, start_date, target=dimension_to_plot, freq=None, save_path=fig_path,probabilities = probabilities, pkl_path_name=f"{dimension_to_plot}")

    if cfg.model.compute_flops:
        from tsExperiments.computation_flops.flops_computation import count_flops_for_predictions

        log.info("Computing FLOPs")
        if (
            hasattr(predictor, "prediction_net")
            and hasattr(predictor.prediction_net, "model")
            and hasattr(predictor.prediction_net.model, "num_parallel_samples")
        ):
            log.info(
                f"Setting the num parallel samples to {cfg.model.params.num_parallel_samples}"
            )
            predictor.prediction_net.model.num_parallel_samples = (
                cfg.model.params.num_parallel_samples
            )

        prediction_flops, total_flops = count_flops_for_predictions(
            predictor, dataset_test, model_name=cfg.model.name
        )

        for logg in logger:
            if hasattr(logg, "log_metrics"):
                logg.log_metrics(
                    {"prediction_flops": prediction_flops, "total_flops": total_flops}
                )
            else:
                log.info(f"Logger {logger} does not have a log_metrics method.")
        log.info("FLOPs computed")
        log.info(f"Prediction FLOPs: {prediction_flops}, Total FLOPs: {total_flops}")

    log.info("Finished experiment with run_name: {}".format(cfg.run_name))


if __name__ == "__main__":
    main()