# tsExperiments/models/project_models/TimePrism/timePrism_estimator.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Iterator, Iterable
import rootutils
from tqdm import tqdm
# Setup project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import DataLoader, as_stacked_batches
from gluonts.itertools import Cyclic, select
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.forecast_generator import ForecastGenerator, to_numpy
from gluonts.time_feature import TimeFeature
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpandDimArray,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    SetFieldIfNotPresent,
    RenameFields,
    TargetDimIndicator,
    Transformation,
    VstackFeatures,
)

from tsExperiments.utils.utils import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)
from .lighting_prism import TimePrismLightning
from tsExperiments.Estimator import PyTorchLightningEstimator

from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

def make_predictions(prediction_net, inputs: dict):
    return prediction_net(**inputs)

@to_numpy.register(torch.Tensor)
def _(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()

class TimePrismSampleForecastGenerator(ForecastGenerator):
    """
    A simple forecast generator for TimePrism.
    It takes the generated samples and wraps them in a SampleForecast object.
    """
    @validated()
    def __init__(self):
        pass

    def __call__(
        self,
        inference_data_loader: DataLoader,
        prediction_net,
        input_names: List[str],
        output_transform: Optional[Callable],
        **kwargs,
    ) -> Iterator[Forecast]:

        for batch in inference_data_loader:
            inputs = select(input_names, batch, ignore_missing=True)
            outputs = to_numpy(make_predictions(prediction_net, inputs))
            if output_transform is not None:
                outputs = output_transform(batch, outputs)
            
            if outputs.shape[-1] == batch['past_target_cdf'].shape[-1] + 1:
                    outputs = outputs[..., :-1]

            i = -1
            for i, output in enumerate(outputs):
                yield SampleForecast(
                    output,
                    start_date=batch[FieldName.FORECAST_START][i],
                    item_id=(
                        batch[FieldName.ITEM_ID][i]
                        if FieldName.ITEM_ID in batch
                        else None
                    ),
                    info=batch["info"][i] if "info" in batch else None,
                )
            assert i + 1 == len(batch[FieldName.FORECAST_START])


class TimePrismEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        # --- All arguments without a default value come first ---
        freq: str,
        prediction_length: int,
        target_dim: int,
        context_length: int,
        num_hypotheses: int,
        individual: bool,
        decomp_kernel_size: int,
        wta_mode_params: dict,
        score_loss_lambda: float,
        num_parallel_samples: int,
        scaling: bool,
        scaler_type: str,
        div_by_std: bool,
        minimum_std: float,
        minimum_std_cst: float,
        default_scale: bool,
        default_scale_cst: bool,
        add_minimum_std: bool,
        
        # --- All arguments with a default value come after ---
        use_full_history: bool = False, # ADDED: The new switch parameter
        pick_incomplete: bool = False,
        use_dynamic_features: bool = True, # ADDED: The new switch parameter
        interaction_rank: int = 4, # ADDED: The new parameter for interaction rank
        embedding_dimension: int = 0,
        time_features: Optional[List[TimeFeature]] = None,
        num_feat_dynamic_real: int = 0, # MODIFIED: Now receives value from YAML
        trainer_kwargs: dict = {},
        data_kwargs: dict = {},
        optim_kwargs: dict = {},
        **kwargs,
    ):
        log.info(f"kwargs (not used in constructor): {kwargs}")

        self.trainer_kwargs = trainer_kwargs
        self.batch_size = data_kwargs["batch_size"]
        self.num_batches_per_epoch = data_kwargs["num_batches_per_epoch"]
        self.num_batches_val_per_epoch = data_kwargs.get("num_batches_val_per_epoch", self.num_batches_per_epoch)
        self.shuffle_buffer_length = data_kwargs.get("shuffle_buffer_length")

        super().__init__(trainer_kwargs=trainer_kwargs)

        # Model parameters
        self.freq = freq
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.context_length = context_length
        self.num_hypotheses = num_hypotheses
        self.individual = individual
        self.decomp_kernel_size = decomp_kernel_size
        self.wta_mode_params = wta_mode_params
        self.score_loss_lambda = score_loss_lambda
        self.num_parallel_samples = num_parallel_samples
        
        # Data processing and scaling parameters
        self.scaling = scaling
        self.scaler_type = scaler_type
        self.div_by_std = div_by_std
        self.minimum_std = minimum_std
        self.minimum_std_cst = minimum_std_cst
        self.default_scale = default_scale
        self.default_scale_cst = default_scale_cst
        self.add_minimum_std = add_minimum_std
        self.pick_incomplete = pick_incomplete
        self.time_features = time_features or fourier_time_features_from_frequency(self.freq)
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.optim_kwargs = optim_kwargs
        self.use_full_history = use_full_history # ADDED: Store the flag
        self.use_dynamic_features = use_dynamic_features # ADDED: Store the flag
        self.interaction_rank = interaction_rank # ADDED: Store the new parameter
        self.embedding_dimension = embedding_dimension

        # MODIFIED: history_length is the total lookback window the model will receive.
        # This is calculated to be the same length as timeMCL's input.
        lags_seq = lags_for_fourier_time_features_from_frequency(freq_str=freq)
        self.history_length = self.context_length + max(lags_seq)
        log.info(f"Total history length for the model will be: {self.history_length}")


        self.full_input_names = [
            "target_dimension_indicator",
            "past_target_cdf",
            "past_observed_values",
            "past_is_pad",
            "future_time_feat",
            "past_time_feat",
            "future_target_cdf",
            "future_observed_values",
        ]

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        self.val_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )
    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=2),
                ExpandDimArray(field=FieldName.TARGET, axis=None),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                    h_stack=False,
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def _create_instance_splitter(self, module, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.val_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
        ) + RenameFields(
            {
                f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
            }
        )

    def create_training_data_loader(self, data: Dataset, module, **kwargs) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(data, is_train=True)
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=self.shuffle_buffer_length,
            field_names=self.full_input_names,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )
        
    def create_validation_data_loader(self, data: Dataset, module, **kwargs) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "validation").apply(data, is_train=True)
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=self.shuffle_buffer_length,
            field_names=self.full_input_names,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_val_per_epoch,
        )

    def create_lightning_module(self) -> nn.Module:
        return TimePrismLightning(
            model_kwargs={
                "num_hypotheses": self.num_hypotheses,
                "individual": self.individual,
                "decomp_kernel_size": self.decomp_kernel_size,
                "wta_mode_params": self.wta_mode_params,
                "score_loss_lambda": self.score_loss_lambda,
                "context_length": self.context_length, # Used for scaler
                "prediction_length": self.prediction_length,
                "target_dim": self.target_dim,
                "num_parallel_samples": self.num_parallel_samples,
                "scaling": self.scaling,
                "scaler_type": self.scaler_type,
                "div_by_std": self.div_by_std,
                "minimum_std": self.minimum_std,
                "minimum_std_cst": self.minimum_std_cst,
                "default_scale": self.default_scale,
                "default_scale_cst": self.default_scale_cst,
                "add_minimum_std": self.add_minimum_std,
                # MODIFIED: Pass the full history_length to the network.
                # This will become the new `seq_len` for the core model.
                "history_length": self.history_length,
                "use_full_history": self.use_full_history,
                "num_feat_dynamic_real": self.num_feat_dynamic_real,
                "use_dynamic_features": self.use_dynamic_features,
                "interaction_rank": self.interaction_rank, # ADDED: Pass the new parameter
                "embedding_dimension": self.embedding_dimension

            },
            optim_kwargs=self.optim_kwargs,
        )
        
    def create_predictor(
            self, transformation: Transformation, module
        ) -> PyTorchPredictor:
            prediction_splitter = self._create_instance_splitter(module, "test")
            
            # MODIFIED: Add 'past_time_feat' to the list of inputs required for prediction.
            # This ensures the predictor provides this tensor to the model's forward method.
            prediction_input_names = [
                "past_target_cdf", 
                "past_observed_values",
                "past_time_feat"
            ]

            return PyTorchPredictor(
                input_transform=transformation + prediction_splitter,
                input_names=prediction_input_names,
                prediction_net=module,
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                device="auto",
                forecast_generator=TimePrismSampleForecastGenerator(),
            )

def run_timeprism_evaluation(
    predictor: PyTorchPredictor, 
    estimator: TimePrismEstimator, 
    dataset_test: Dataset
) -> tuple[List[dict], List[dict]]:
    """
    Runs a dedicated evaluation loop for TimePrism.

    This function iterates through the test dataset, applies the necessary
    transformations to get model inputs, calls the model's dedicated
    evaluation method to get all hypotheses and their probabilities, and
    returns the raw model outputs alongside the raw test entries for
    downstream processing by the evaluator.

    Args:
        predictor: The trained PyTorchPredictor object.
        estimator: The estimator instance used to create the predictor, needed for transformations.
        dataset_test: The GluonTS test dataset.

    Returns:
        A tuple containing two lists:
        - forecasts_list: A list of dictionaries, each with "hypotheses" and "probabilities".
        - targets_list: A list of the raw, unprocessed test_entry dictionaries.
    """
    
    forecasts_list = []
    targets_list = []
    
    # Get the exact transformation and prediction splitter from the estimator
    transformation = estimator.create_transformation()
    prediction_splitter = estimator._create_instance_splitter(predictor.prediction_net, "test")
    full_pipeline = transformation + prediction_splitter
    
    # Manually iterate and process each entry in the test set
    for test_entry in tqdm(iter(dataset_test), total=len(dataset_test), desc="TimePrism Evaluation"):
        # The .apply method takes an iterable; we pass a list with a single entry.
        # It yields a generator, from which we take the single processed dictionary.
        data_t = next(iter(full_pipeline.apply([test_entry], is_train=False)))

        # Extract tensors and move them to the correct device
        device = predictor.prediction_net.device
        past_target_cdf = torch.tensor(data_t['past_target_cdf'], dtype=torch.float32).unsqueeze(0).to(device)
        past_observed_values = torch.tensor(data_t['past_observed_values'], dtype=torch.float32).unsqueeze(0).to(device)
        past_time_feat = torch.tensor(data_t['past_time_feat'], dtype=torch.float32).unsqueeze(0).to(device)
        target_dim_indicator = torch.tensor(data_t['target_dimension_indicator'], dtype=torch.long).unsqueeze(0).to(device)

        # Call the dedicated evaluation forward pass
        with torch.no_grad():
            hypotheses, probabilities = predictor.prediction_net.forward_for_evaluation(
                past_target_cdf=past_target_cdf,
                past_observed_values=past_observed_values,
                past_time_feat=past_time_feat,
                target_dimension_indicator=target_dim_indicator

            )
        
        # Store raw model outputs
        forecasts_list.append({
            "hypotheses": hypotheses.cpu().numpy(),
            "probabilities": probabilities.cpu().numpy()
        })
        
        # Store the entire raw entry for the evaluator to process
        targets_list.append(test_entry)

    return forecasts_list, targets_list