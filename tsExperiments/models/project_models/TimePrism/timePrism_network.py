# tsExperiments/models/project_models/timePrism/timePrism_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import Tensor
from typing import List, Optional, Tuple
from argparse import Namespace

from gluonts.core.component import validated
# FIX: Corrected import path according to the new file structure
from .DLinear_Encoder import DLinear_Encoder
from tsExperiments.data_and_transformation import (
    MeanScaler,
    NOPScaler,
    MeanStdScaler,
    CenteredMeanScaler,
)
from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)
class ResBlock(nn.Module):
    """
    A simple residual block with two linear layers.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.layer1(x))
        out = self.layer2(out)
        out += residual
        return F.relu(out)

class TimePrismCore(nn.Module):
    """
    The definitive, DLinear-inspired probabilistic model featuring "Trend-Guided Empirical Shift".
    This is the core implementation of the timePrism model. It operates on NORMALIZED data.
    """
    def __init__(self, configs):
        super(TimePrismCore, self).__init__()
        # --- Basic model and data configuration ---
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        # REMOVED: The 'individual' flag is no longer supported.
        self.num_feat_dynamic_real = getattr(configs, 'num_feat_dynamic_real', 0)
        self.use_dynamic_features = configs.use_dynamic_features
        # ADDED: New hyperparameter for the rank of the interaction matrix
        self.interaction_rank = getattr(configs, 'interaction_rank', 8)
        self.embedding_dim = getattr(configs, 'embedding_dimension', 0)
        self.channel_embedding = None
        
        # Only create the embedding layer if the dimension is greater than 0.
        if self.embedding_dim > 0:
            log.info(f"Channel embedding is ENABLED with dimension {self.embedding_dim}.")
            self.channel_embedding = nn.Embedding(
                num_embeddings=self.enc_in,
                embedding_dim=self.embedding_dim
            )
        else:
            log.info("Channel embedding is DISABLED.")
        # --- Intelligent Factorization of Hypotheses ---
        total_hypotheses = getattr(configs, 'num_hypotheses', 1)
        if total_hypotheses <= 0: total_hypotheses = 1
        
        n_trend = int(total_hypotheses**0.5)
        while n_trend > 0 and total_hypotheses % n_trend != 0:
            n_trend -= 1
        if n_trend == 0: n_trend = 1
        
        self.n_hypotheses_trend = n_trend
        self.n_hypotheses_seasonal = total_hypotheses // n_trend
        self.total_components = total_hypotheses
        log.info(f"Factorized total_hypotheses={total_hypotheses} into N={self.n_hypotheses_trend} (trend) and K={self.n_hypotheses_seasonal} (seasonal)")

        # --- Loss Function Configuration ---
        wta_params = getattr(configs, 'wta_mode_params')
        self.wta_mode = wta_params.get('mode', 'wta')
        self.r_wta_epsilon = wta_params.get('epsilon', 0.1)
        self.wta_temperature = wta_params.get('temperature_ini', 1.0)
        self.score_loss_lambda = getattr(configs, 'score_loss_lambda', 1.0)

        # --- Lightweight Decompositional Encoder ---
        self.ts_encoder = DLinear_Encoder(
            decomp_kernel_size=getattr(configs, 'decomp_kernel_size', 25)
        )
        features_to_add = self.num_feat_dynamic_real if self.use_dynamic_features else 0
        log.info(f"Number of dynamic features added to heads: {features_to_add}")
        
        trend_seasonal_input_dim = self.seq_len + features_to_add + self.embedding_dim
        pi_decoder_input_dim = self.seq_len + features_to_add + self.embedding_dim

        # --- MODIFIED: Replaced the single pi_decoder with four specialized heads for low-rank interaction ---
        # This unified head will be the input for the specialized heads below
        pi_hidden_dim = 512
        # self.pi_base = nn.Sequential(
        #     nn.Linear(pi_decoder_input_dim, pi_hidden_dim),
        #     nn.ReLU()
        #     # ResBlock(hidden_dim=pi_hidden_dim),
        #     # ResBlock(hidden_dim=pi_hidden_dim)
        # )
        # self.pi_base = nn.Sequential(
        #     nn.Linear(pi_decoder_input_dim, pi_hidden_dim), nn.ReLU()
        # )
        self.pi_decoder = nn.Linear(pi_decoder_input_dim, self.total_components)


        # # Head for Trend main effect P(t|X)
        # self.pi_head_trend = nn.Linear(pi_hidden_dim, self.n_hypotheses_trend)
        # # Head for Seasonal main effect P(s|X)
        # self.pi_head_seasonal = nn.Linear(pi_hidden_dim, self.n_hypotheses_seasonal)
        # # Head for Trend interaction embedding U
        # self.pi_head_U = nn.Linear(pi_hidden_dim, self.n_hypotheses_trend * self.interaction_rank)
        # # Head for Seasonal interaction embedding V
        # self.pi_head_V = nn.Linear(pi_hidden_dim, self.n_hypotheses_seasonal * self.interaction_rank)





        self.trend_head = nn.Linear(trend_seasonal_input_dim, self.pred_len * self.n_hypotheses_trend)
        self.seasonal_head = nn.Linear(trend_seasonal_input_dim, self.pred_len * self.n_hypotheses_seasonal)

        self.last_pi_logits = self.last_trends = self.last_seasonals = None

# In tsExperiments/models/project_models/timePrism/timePrism_network.py
# Inside the TimePrismCore class

    def forward(self, x: Tensor, time_feat: Tensor, **kwargs) -> Tensor:
        # The channel indices are passed via kwargs from the LightningModule
        target_dim_indicator = kwargs["target_dimension_indicator"]

        # --- Step 1: Decompose and Prepare Inputs ---
        seasonal_init, trend_init = self.ts_encoder(x)
        seasonal_permuted = seasonal_init.permute(0, 2, 1)
        trend_permuted = trend_init.permute(0, 2, 1)
        x_permuted = x.permute(0, 2, 1)
        B, C, T_in = trend_permuted.shape
        time_feat_summary = time_feat.mean(dim=1)

        # --- Step 2: Conditionally prepare inputs with embeddings ---
        # Prepare base inputs without embeddings first.
        pi_input_base = x_permuted
        trend_input_base = trend_permuted
        seasonal_input_base = seasonal_permuted
        
        if self.use_dynamic_features:
            time_feat_summary_expanded = time_feat_summary.unsqueeze(1).expand(-1, C, -1)
            pi_input_base = torch.cat([x_permuted, time_feat_summary_expanded], dim=2)
            trend_input_base = torch.cat([trend_permuted, time_feat_summary_expanded], dim=2)
            seasonal_input_base = torch.cat([seasonal_permuted, time_feat_summary_expanded], dim=2)

        # Conditionally concatenate embeddings if the embedding layer exists.
        if self.channel_embedding is not None:
            # Get embedding vector, shape (B, C, embedding_dim)
            channel_embeds = self.channel_embedding(target_dim_indicator)
            
            # Inject embeddings into all relevant inputs by concatenating them.
            pi_input = torch.cat([pi_input_base, channel_embeds], dim=-1)
            trend_input = torch.cat([trend_input_base, channel_embeds], dim=-1)
            seasonal_input = torch.cat([seasonal_input_base, channel_embeds], dim=-1)
        else:
            # If embedding is disabled, use the base inputs directly.
            pi_input = pi_input_base
            trend_input = trend_input_base
            seasonal_input = seasonal_input_base

        # --- Step 3: Calculate Logits ---
        # The pi_decoder now receives an input that might contain channel embeddings.
        pi_logits_flat = self.pi_decoder(pi_input) # Shape: (B, C, N*K)
        # Reshape the flat logits into the (N, K) structure for compatibility with loss and inference
        pi_logits = pi_logits_flat.view(B, C, self.n_hypotheses_trend, self.n_hypotheses_seasonal)
        
        # Cache the final logits
        self.last_pi_logits = pi_logits

        # --- Step 4: Handle Hypothesis Generation and Final Output ---
        # The trend and seasonal heads now receive inputs that might contain channel embeddings.
        trends_flat = self.trend_head(trend_input)
        seasonals_flat = self.seasonal_head(seasonal_input)
        self.last_trends = trends_flat.view(B, C, self.pred_len, self.n_hypotheses_trend).permute(0, 3, 2, 1)
        self.last_seasonals = seasonals_flat.view(B, C, self.pred_len, self.n_hypotheses_seasonal).permute(0, 3, 2, 1)

        if self.training:
            output = torch.mean(self.last_trends, dim=1) + torch.mean(self.last_seasonals, dim=1)
            return output
        
        # --- INFERENCE PATH (Validation or Test) ---
        pi_logits_flat_per_channel = pi_logits.flatten(2, 3)
        best_comp_indices = torch.argmax(pi_logits_flat_per_channel, dim=2)
        best_k_idx = best_comp_indices % self.n_hypotheses_seasonal
        best_n_idx = best_comp_indices // self.n_hypotheses_seasonal


        seasonals_permuted = self.last_seasonals.permute(0, 3, 2, 1)
        index_seasonal = best_k_idx.view(B, C, 1, 1).expand(-1, -1, self.pred_len, 1)
        best_seasonal = torch.gather(seasonals_permuted, 3, index_seasonal).squeeze(-1)
        
        trends_permuted = self.last_trends.permute(0, 3, 2, 1)
        index_trend = best_n_idx.view(B, C, 1, 1).expand(-1, -1, self.pred_len, 1)
        associated_trend = torch.gather(trends_permuted, 3, index_trend).squeeze(-1)
        output = (best_seasonal + associated_trend).permute(0, 2, 1)
        return output

    def get_loss(self, gt_pred_normalized: Tensor) -> Tensor:
        # This function's logic remains entirely unchanged
        trends, seasonals, pi_logits = self.last_trends, self.last_seasonals, self.last_pi_logits
        if trends is None: return torch.tensor(0.0, device=gt_pred_normalized.device)
        
        B, N, T, C = trends.shape
        
        gt_expanded = gt_pred_normalized.unsqueeze(1).unsqueeze(1)

        forecasts = trends.unsqueeze(2) + seasonals.unsqueeze(1)
        squared_errors = torch.pow(gt_expanded - forecasts, 2)
        
        loss_per_hyp_channel = squared_errors.sum(dim=3)
        loss_per_channel_flat = loss_per_hyp_channel.permute(0, 3, 1, 2).flatten(2, 3)
        
        winner_losses_per_channel, winner_indices_per_channel = torch.min(loss_per_channel_flat, dim=2)
        
        if self.wta_mode == 'wta':
            reconstruction_loss = winner_losses_per_channel.mean()
        
        elif self.wta_mode == 'relaxed-wta':
            if self.total_components > 1:
                total_loss_sum_per_channel = loss_per_channel_flat.sum(dim=2)
                losers_loss_sum_per_channel = total_loss_sum_per_channel - winner_losses_per_channel
                reconstruction_loss = ((1 - self.r_wta_epsilon) * winner_losses_per_channel.mean() +
                                       (self.r_wta_epsilon / (self.total_components - 1)) * losers_loss_sum_per_channel.mean())
            else:
                reconstruction_loss = winner_losses_per_channel.mean()

        elif self.wta_mode == 'awta':
            temperature = self.wta_temperature
            amcl_weights = torch.softmax(-loss_per_channel_flat / temperature, dim=2).detach()
            reconstruction_loss = (amcl_weights * loss_per_channel_flat).sum(dim=2).mean()

        else:
            raise ValueError(f"Unsupported wta_mode: {self.wta_mode}")
        
        pi_logits_flat = pi_logits.permute(0, 1, 2, 3).flatten(2, 3).reshape(B * C, self.total_components)
        winner_indices_flat = winner_indices_per_channel.reshape(-1)
        score_loss = F.cross_entropy(pi_logits_flat, winner_indices_flat)

        final_loss = reconstruction_loss + self.score_loss_lambda * score_loss
        return final_loss

    def sample_many(self, x: Tensor, time_feat: Tensor, n_samples: int = 10, **kwargs) -> Tensor:
        self.eval()
        samples = []
        with torch.no_grad():
            self.forward(x, time_feat=time_feat, **kwargs)
            
            pi_logits, trends, seasonals = self.last_pi_logits, self.last_trends, self.last_seasonals
            if trends is None: return x.new_zeros(n_samples, x.shape[0], self.pred_len, self.enc_in)

            pi_logits_flat_per_channel = pi_logits.flatten(2, 3)
            pi_dist = Categorical(logits=pi_logits_flat_per_channel)

            trends_permuted = trends.permute(0, 3, 2, 1)
            seasonals_permuted = seasonals.permute(0, 3, 2, 1)

            for _ in range(n_samples):
                joint_indices = pi_dist.sample()
                n_indices = joint_indices // self.n_hypotheses_seasonal
                k_indices = joint_indices % self.n_hypotheses_seasonal
                
                index_trend = n_indices.view(x.shape[0], self.enc_in, 1, 1).expand(-1, -1, self.pred_len, 1)
                trend_s = torch.gather(trends_permuted, 3, index_trend).squeeze(-1)

                index_seasonal = k_indices.view(x.shape[0], self.enc_in, 1, 1).expand(-1, -1, self.pred_len, 1)
                seasonal_s = torch.gather(seasonals_permuted, 3, index_seasonal).squeeze(-1)

                forecast = (trend_s + seasonal_s).permute(0, 2, 1)
                samples.append(forecast)
        
        return torch.stack(samples, dim=0)

# In tsExperiments/models/project_models/timePrism/timePrism_network.py
# Inside the TimePrismCore class

    def forward_for_evaluation(self, x: Tensor, time_feat: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        A dedicated forward pass for evaluation that returns all hypotheses and their probabilities.
        This avoids the sampling or argmax logic of the standard forward pass.
        It conditionally uses channel embeddings if they are enabled.
        """
        # The channel indices are passed via kwargs from the LightningModule
        target_dim_indicator = kwargs["target_dimension_indicator"]

        # --- Step 1: Decompose and Prepare Inputs ---
        seasonal_init, trend_init = self.ts_encoder(x)
        seasonal_permuted = seasonal_init.permute(0, 2, 1)
        trend_permuted = trend_init.permute(0, 2, 1)
        x_permuted = x.permute(0, 2, 1)
        B, C, T_in = trend_permuted.shape
        time_feat_summary = time_feat.mean(dim=1)

        # --- Step 2: Conditionally prepare inputs with embeddings ---
        # Prepare base inputs without embeddings first.
        pi_input_base = x_permuted
        trend_input_base = trend_permuted
        seasonal_input_base = seasonal_permuted
        
        if self.use_dynamic_features:
            time_feat_summary_expanded = time_feat_summary.unsqueeze(1).expand(-1, C, -1)
            pi_input_base = torch.cat([x_permuted, time_feat_summary_expanded], dim=2)
            trend_input_base = torch.cat([trend_permuted, time_feat_summary_expanded], dim=2)
            seasonal_input_base = torch.cat([seasonal_permuted, time_feat_summary_expanded], dim=2)

        # Conditionally concatenate embeddings if the embedding layer exists.
        if self.channel_embedding is not None:
            # Get embedding vector, shape (B, C, embedding_dim)
            channel_embeds = self.channel_embedding(target_dim_indicator)
            
            # Inject embeddings into all relevant inputs by concatenating them.
            pi_input = torch.cat([pi_input_base, channel_embeds], dim=-1)
            trend_input = torch.cat([trend_input_base, channel_embeds], dim=-1)
            seasonal_input = torch.cat([seasonal_input_base, channel_embeds], dim=-1)
        else:
            # If embedding is disabled, use the base inputs directly.
            pi_input = pi_input_base
            trend_input = trend_input_base
            seasonal_input = seasonal_input_base

        # --- Step 3: Calculate Logits ---
        pi_logits_flat = self.pi_decoder(pi_input) # Shape: (B, C, N*K)
        pi_logits = pi_logits_flat.view(B, C, self.n_hypotheses_trend, self.n_hypotheses_seasonal)

        # --- Step 4: Compute all hypotheses ---
        trends_flat = self.trend_head(trend_input)
        seasonals_flat = self.seasonal_head(seasonal_input)

        trends = trends_flat.view(B, C, self.pred_len, self.n_hypotheses_trend).permute(0, 3, 2, 1)
        seasonals = seasonals_flat.view(B, C, self.pred_len, self.n_hypotheses_seasonal).permute(0, 3, 2, 1)

        # --- Step 5: Combine all hypotheses and compute their probabilities ---
        # This is the memory-intensive step that generates all N*K combinations.
        all_hypotheses = trends.unsqueeze(2) + seasonals.unsqueeze(1) # Shape: (B, N, K, T, C)
        
        # Flatten hypotheses to match the flattened logits structure.
        all_hypotheses_flat = all_hypotheses.flatten(1, 2) # Shape: (B, N*K, T, C)
        
        # Compute probabilities from the calculated logits.
        pi_logits_flat_per_channel = pi_logits.flatten(2, 3) # Shape: (B, C, N*K)
        probabilities = torch.softmax(pi_logits_flat_per_channel, dim=-1) # Shape: (B, C, N*K)
        
        # The shape (B, N*K, T, C) is the correct format needed for denormalization and evaluation.
        all_hypotheses_final = all_hypotheses_flat
        # Permute probabilities to match the (Batch, Samples, Dims) format expected by the evaluator.
        probabilities_final = probabilities.permute(0, 2, 1) # (B, C, N*K) -> (B, N*K, C)

        return all_hypotheses_final, probabilities_final

class TimePrismNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        # TimePrism specific arguments
        num_hypotheses: int,
        individual: bool,
        decomp_kernel_size: int,
        wta_mode_params: dict,
        score_loss_lambda: float,
        
        # GluonTS specific arguments
        context_length: int,
        prediction_length: int,
        target_dim: int,
        num_parallel_samples: int,
        
        # MODIFIED: history_length is a key parameter for seq_len
        history_length: int,

        # ADDED: A switch to control the input sequence length
        use_full_history: bool,
        
        # ADDED: Number of dynamic real features to expect
        num_feat_dynamic_real: int,
        use_dynamic_features: bool, # ADDED
        interaction_rank: int,
        embedding_dimension: int,
        # Scaling arguments
        scaling: bool,
        scaler_type: str,
        div_by_std: bool,
        minimum_std: float,
        minimum_std_cst: float,
        default_scale: bool,
        default_scale_cst: bool,
        add_minimum_std: bool,
        
        **kwargs,
    ):
        super().__init__()
        # NOTE: context_length is kept for the scaler, which operates on a smaller, recent window
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_parallel_samples = num_parallel_samples
        self.scaler_type = scaler_type
        self.div_by_std = div_by_std
        
        self.history_length = history_length
        # ADDED: Store the flag for use in forward/loss methods
        self.use_full_history = use_full_history
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.use_dynamic_features = use_dynamic_features # ADDED
        self.interaction_rank = interaction_rank # ADDED
        self.embedding_dimension = embedding_dimension # ADDED

        # Conditionally determine the sequence length for the core model's architecture
        if self.use_full_history:
            core_seq_len = self.history_length
            log.info(f"Model configured to use FULL history length for decomposition: {core_seq_len}")
        else:
            core_seq_len = self.context_length
            log.info(f"Model configured to use SHORT history length (context_length) for decomposition: {core_seq_len}")

        # Create a simple namespace object to hold the configuration
        model_configs = Namespace(
            # MODIFIED: The core model's sequence length is now conditionally set
            seq_len=core_seq_len,
            pred_len=prediction_length,
            enc_in=target_dim,
            num_hypotheses=num_hypotheses,
            individual=individual,
            decomp_kernel_size=decomp_kernel_size,
            wta_mode_params=wta_mode_params,
            score_loss_lambda=score_loss_lambda,
            num_feat_dynamic_real=num_feat_dynamic_real,
            use_dynamic_features=use_dynamic_features, 
            interaction_rank=interaction_rank,
            embedding_dimension=embedding_dimension,
        )

        self.model = TimePrismCore(configs=model_configs)

        self.scaling = scaling
        if self.scaling:
            if self.scaler_type == "mean":
                self.scaler = MeanScaler(keepdim=True)
            elif self.scaler_type == "nops":
                self.scaler = NOPScaler(keepdim=True)
            elif self.scaler_type == "centered_mean":
                self.scaler = CenteredMeanScaler(keepdim=True)
            elif self.scaler_type == "mean_std":
                self.scaler = MeanStdScaler(
                    minimum_std=minimum_std,
                    minimum_std_cst=minimum_std_cst,
                    default_scale=default_scale,
                    default_scale_cst=default_scale_cst,
                    add_minimum_std=add_minimum_std,
                    keepdim=True,
                )
            else:
                raise ValueError(f"Invalid scaler type: {self.scaler_type}")

    def _compute_scale(self, past_target: torch.Tensor, past_observed: torch.Tensor) -> dict:
        # NOTE: The scaler still operates on the most recent `context_length` part of the history
        if self.scaler_type in ["mean_std", "centered_mean"]:
            _, mean, std = self.scaler(
                past_target[:, -self.context_length :, ...],
                past_observed[:, -self.context_length :, ...],
            )
            return {"mean": mean, "std": std}
        elif self.scaler_type in ["mean", "nops"]:
            _, scale = self.scaler(
                past_target[:, -self.context_length :, ...],
                past_observed[:, -self.context_length :, ...],
            )
            return {"scale": scale}
        else:
            return {}

    def _normalize_tensor(self, tensor: torch.Tensor, scale_params: dict) -> torch.Tensor:
        if not self.scaling or not scale_params:
            return tensor
            
        if self.scaler_type in ["mean_std", "centered_mean"]:
            mean = scale_params["mean"]
            std = scale_params["std"]
            if self.div_by_std:
                return (tensor - mean) / std
            else:
                return tensor - mean
        elif self.scaler_type in ["mean", "nops"]:
            return tensor / scale_params["scale"]
        return tensor

    def _denormalize_tensor(self, tensor: torch.Tensor, scale_params: dict) -> torch.Tensor:
        if not self.scaling or not scale_params:
            return tensor

        if self.scaler_type in ["mean_std", "centered_mean"]:
            mean = scale_params["mean"]
            std = scale_params["std"]
            if self.div_by_std:
                while mean.dim() < tensor.dim():
                    mean = mean.unsqueeze(0)
                    std = std.unsqueeze(0)
                return tensor * std + mean
            else:
                while mean.dim() < tensor.dim():
                    mean = mean.unsqueeze(0)
                return tensor + mean
        elif self.scaler_type in ["mean", "nops"]:
            scale = scale_params["scale"]
            while scale.dim() < tensor.dim():
                scale = scale.unsqueeze(0)
            return tensor * scale
        return tensor

    def loss(
        self,
        past_target_cdf: torch.Tensor,
        future_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        
        scale_params = self._compute_scale(past_target_cdf, past_observed_values)
        
        # Conditionally select the input tensor for the model based on the flag.
        # The data loader ALWAYS provides the full history_length.
        if self.use_full_history:
            # Use the entire historical sequence
            model_input_tensor = past_target_cdf
            # Select the corresponding time features
            time_feat_tensor = past_time_feat
        else:
            # Slice the history to use only the most recent context_length part
            model_input_tensor = past_target_cdf[:, -self.context_length :, :]
            # Select the corresponding time features
            time_feat_tensor = past_time_feat[:, -self.context_length :, :]
        
        x_in_normalized = self._normalize_tensor(model_input_tensor, scale_params)
        
        # The ground truth for the loss is still only the prediction_length part
        gt_normalized = self._normalize_tensor(
            torch.cat((past_target_cdf, future_target_cdf), dim=1),
            scale_params
        )
        gt_pred_normalized = gt_normalized[:, -self.prediction_length:, :]
        # if torch.isnan(gt_pred_normalized).any():
        #     print("!!! WARNING: NaN found in normalized ground truth !!!")
        # print(f"Stats for normalized GT: mean={gt_pred_normalized.mean().item():.2f}, std={gt_pred_normalized.std().item():.2f}, max={gt_pred_normalized.max().item():.2f}, min={gt_pred_normalized.min().item():.2f}")
        # The model's forward pass is called with the (potentially sliced) input and time features
        self.model(x_in_normalized, time_feat=time_feat_tensor, **kwargs)
        final_loss = self.model.get_loss(gt_pred_normalized)
        
        return (final_loss, None, None, None, None)

    def forward(
        self,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        scale_params = self._compute_scale(past_target_cdf, past_observed_values)
        
        # Conditionally select the input tensor for the model based on the flag
        if self.use_full_history:
            # Use the entire historical sequence
            model_input_tensor = past_target_cdf
            # Select the corresponding time features
            time_feat_tensor = past_time_feat
        else:
            # Slice the history to use only the most recent context_length part
            model_input_tensor = past_target_cdf[:, -self.context_length :, :]
            # Select the corresponding time features
            time_feat_tensor = past_time_feat[:, -self.context_length :, :]

        x_in_normalized = self._normalize_tensor(model_input_tensor, scale_params)
        
        # sample_many now operates on the results of the forward pass which used the (potentially sliced) sequence and time features
        # First, call forward to populate the caches
        self.model(x_in_normalized, time_feat=time_feat_tensor, **kwargs)
        # Then, call sample_many which uses the cached values
        normalized_samples = self.model.sample_many(
            x_in_normalized, 
            time_feat=time_feat_tensor, 
            n_samples=self.num_parallel_samples,
            **kwargs 
        )
        denormalized_samples = self._denormalize_tensor(normalized_samples, scale_params)

        samples = denormalized_samples.permute(1, 0, 2, 3)

        B, K, T, C = samples.shape
        dummy_scores = torch.ones(B, K, T, 1, device=samples.device)
        
        samples_with_scores = torch.cat([samples, dummy_scores], dim=-1)

        return samples_with_scores

    def forward_for_evaluation(
        self,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A dedicated forward pass for evaluation that returns denormalized hypotheses and their probabilities.
        """
        scale_params = self._compute_scale(past_target_cdf, past_observed_values)
        
        if self.use_full_history:
            model_input_tensor = past_target_cdf
            time_feat_tensor = past_time_feat
        else:
            model_input_tensor = past_target_cdf[:, -self.context_length :, :]
            time_feat_tensor = past_time_feat[:, -self.context_length :, :]

        x_in_normalized = self._normalize_tensor(model_input_tensor, scale_params)
        
        # Call the new core model method
        normalized_hypotheses, probabilities = self.model.forward_for_evaluation(
            x_in_normalized, time_feat=time_feat_tensor, **kwargs
        )
        
        # Denormalize the hypotheses
        denormalized_hypotheses = self._denormalize_tensor(normalized_hypotheses, scale_params)
        
        # The output shapes are expected to be (B, K, T, C) and (B, K, C)
        # B=1 for evaluation, so we squeeze it.
        # This returns all N*K hypotheses and their corresponding probabilities.
        return denormalized_hypotheses.squeeze(0), probabilities.squeeze(0)