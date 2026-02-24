# tsExperiments/computation_flops/flops_computation.py (Definitive Final Version)

import rootutils
import os, sys
import torch
from fvcore.nn import FlopCountAnalysis
from typing import Tuple
import torch.nn as nn

# Setup project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))


def count_flops_for_predictions(
    predictor, dataset_test, num_samples=10, model_name=None
):
    """
    Counts FLOPs for the prediction process. This function now dispatches
    to the correct logic based on the model_name.
    """

    # --- New, dedicated logic for TimePrism using a clean dummy input ---
    def get_prediction_flops_timeprism():
        # --- FIX: Define a temporary wrapper to handle the complex signature ---
        # This wrapper provides a simple forward signature that fvcore can trace,
        # and then calls the actual model with the correct keyword arguments.
        class ModelWrapperForFlops(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(
                self, 
                past_target_cdf, 
                past_observed_values, 
                past_time_feat, 
                target_dimension_indicator
            ):
                # Inside the wrapper, we call the REAL model's forward pass
                # using the keyword arguments it expects.
                return self.model(
                    past_target_cdf=past_target_cdf,
                    past_observed_values=past_observed_values,
                    past_time_feat=past_time_feat,
                    target_dimension_indicator=target_dimension_indicator,
                )

        with torch.no_grad():
            # The model being wrapped is the LightningModule
            model = predictor.prediction_net
            model.eval()

            # Define shapes based on the model's actual configuration
            batch_size = 1
            history_length = model.model.context_length 
            target_dim = model.model.target_dim
            num_feat_dynamic_real = model.model.num_feat_dynamic_real
            
            device = next(model.parameters()).device

            # Create a tuple of tensors. The order MUST match the wrapper's forward signature.
            inputs_tuple = (
                torch.ones(batch_size, history_length, target_dim, device=device, dtype=torch.float32),
                torch.ones(batch_size, history_length, target_dim, device=device, dtype=torch.float32),
                torch.ones(batch_size, history_length, num_feat_dynamic_real, device=device, dtype=torch.float32),
                torch.zeros(batch_size, target_dim, device=device, dtype=torch.long),
            )
            
            # --- FIX: Trace the WRAPPER, not the original model ---
            # The wrapper has a simple signature that FlopCountAnalysis can handle.
            wrapped_model = ModelWrapperForFlops(model)
            flops = FlopCountAnalysis(wrapped_model, inputs_tuple)
            
            total_flops = flops.total()
            return total_flops

    # --- Ancestor logic for TransformerTempFlow ---
    def get_prediction_flops_trf_tempflow():
        with torch.no_grad():
            if hasattr(predictor, "prediction_net"):
                model = predictor.prediction_net
            else:
                model = predictor.model

            model.eval()
            model.model.eval()
            batch_size = 1
            target_dim = model.model.target_dim
            history_length = model.model.history_length
            num_feat_dynamic_real = model.model.num_feat_dynamic_real
            prediction_length = model.model.prediction_length
            device = next(model.parameters()).device
            dummy_input = {
                "target_dimension_indicator": torch.zeros(
                    batch_size, target_dim, device=device, dtype=torch.long
                ),
                "past_time_feat": torch.ones(
                    batch_size, history_length, num_feat_dynamic_real, device=device
                ),
                "past_target_cdf": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_observed_values": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_is_pad": torch.ones(batch_size, history_length, device=device),
                "future_time_feat": torch.ones(
                    batch_size, prediction_length, num_feat_dynamic_real, device=device
                ),
                "future_target_cdf": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
                "future_observed_values": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
            }
            inputs = (
                dummy_input["target_dimension_indicator"],
                dummy_input["past_time_feat"],
                dummy_input["past_target_cdf"],
                dummy_input["past_observed_values"],
                dummy_input["past_is_pad"],
                dummy_input["future_time_feat"],
                dummy_input["future_target_cdf"],
                dummy_input["future_observed_values"],
            )
            flops = FlopCountAnalysis(model.model, inputs)
            total_flops = flops.total()
            return total_flops

    # --- Ancestor logic for Tactis ---
    def get_prediction_flops_tactis():
        with torch.no_grad():
            if hasattr(predictor, "prediction_net"):
                model = predictor.prediction_net
            else:
                model = predictor.model
            batch_size = 1
            target_dim = model.model.target_dim
            context_length = model.model.context_length
            prediction_length = model.model.prediction_length
            device = next(model.parameters()).device
            dummy_input = {
                "past_target_norm": torch.ones(
                    batch_size, context_length, target_dim, device=device
                ),
                "future_target_norm": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
            }
            inputs = (
                dummy_input["past_target_norm"],
                dummy_input["future_target_norm"],
            )
            flops = FlopCountAnalysis(model.model, inputs)
            total_flops = flops.total()
            return total_flops

    # --- Ancestor logic for all other default models ---
    def get_prediction_flops():
        with torch.no_grad():
            if hasattr(predictor, "prediction_net"):
                model = predictor.prediction_net
            else:
                model = predictor.model
            model.eval()
            model.model.eval()
            batch_size = 1
            target_dim = model.model.target_dim
            history_length = model.model.history_length
            num_feat_dynamic_real = model.model.num_feat_dynamic_real
            prediction_length = model.model.prediction_length
            device = next(model.parameters()).device
            dummy_input = {
                "target_dimension_indicator": torch.zeros(
                    batch_size, target_dim, device=device, dtype=torch.long
                ),
                "past_target": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_observed_values": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_target_cdf": torch.ones(
                    batch_size, history_length, target_dim, device=device
                ),
                "past_is_pad": torch.ones(batch_size, history_length, device=device),
                "future_time_feat": torch.ones(
                    batch_size, prediction_length, num_feat_dynamic_real, device=device
                ),
                "past_time_feat": torch.ones(
                    batch_size, history_length, num_feat_dynamic_real, device=device
                ),
                "future_target_cdf": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
                "future_observed_values": torch.ones(
                    batch_size, prediction_length, target_dim, device=device
                ),
            }
            inputs = (
                dummy_input["target_dimension_indicator"],
                dummy_input["past_target_cdf"],
                dummy_input["past_observed_values"],
                dummy_input["past_is_pad"],
                dummy_input["future_time_feat"],
                dummy_input["past_time_feat"],
                dummy_input["future_target_cdf"],
                dummy_input["future_observed_values"],
            )
            flops = FlopCountAnalysis(model.model, inputs)
            total_flops = flops.total()
            return total_flops

    # --- Main Dispatcher Logic ---
    if model_name == "tactis2":
        prediction_flops = get_prediction_flops_tactis()
    elif model_name == "transformer_tempflow":
        prediction_flops = get_prediction_flops_trf_tempflow()
    elif model_name == "timePrism":
        prediction_flops = get_prediction_flops_timeprism()
    else:
        prediction_flops = get_prediction_flops()

    # Calculate total FLOPs
    total_flops = prediction_flops * len(dataset_test)
    return prediction_flops, total_flops