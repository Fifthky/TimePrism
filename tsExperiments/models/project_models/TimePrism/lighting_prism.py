# tsExperiments/models/project_models/TimePrism/lighting_prism.py

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tsExperiments.models.project_models.TimePrism.timePrism_network import TimePrismNetwork
from gluonts.itertools import select
from gluonts.torch.model.lightning_util import has_validation_loop

class TimePrismLightning(pl.LightningModule):

    def __init__(
        self,
        model_kwargs: dict,
        optim_kwargs: dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_kwargs = model_kwargs
        self.optim_kwargs = optim_kwargs

        self.model = TimePrismNetwork(
            **model_kwargs,
        )

        # Training parameters
        self.lr = self.optim_kwargs["lr"]
        self.weight_decay = self.optim_kwargs["weight_decay"]
        self.patience = self.optim_kwargs["patience"]

    # ADDED: Annealing schedule for 'awta' mode, logic copied from timeMCL
    def on_train_epoch_start(self) -> None:
        """
        Lightning hook that is called when a training epoch starts.
        Used to update the temperature for the 'awta' loss.
        """
        wta_params = self.model_kwargs.get("wta_mode_params", {})
        if wta_params.get("mode") != "awta":
            return # Do nothing if not in awta mode

        scheduler_mode = wta_params.get("scheduler_mode", "constant")
        temp_ini = wta_params.get("temperature_ini", 1.0)
        
        if scheduler_mode == "constant":
            temperature = temp_ini
        elif scheduler_mode == "linear":
            temperature = temp_ini - (temp_ini / self.trainer.max_epochs) * self.current_epoch
        elif scheduler_mode == "exponential":
            decay = wta_params.get("temperature_decay", 0.95)
            temperature = temp_ini * (decay ** self.current_epoch)
        else:
            raise ValueError(f"Unknown temperature scheduler mode: {scheduler_mode}")

        # Ensure temperature does not fall below the limit
        temp_lim = wta_params.get("temperature_lim", 1.0e-8)
        temperature = max(temperature, temp_lim)

        # Update the temperature in the core model
        self.model.model.wta_temperature = temperature
        self.log("wta_temperature", temperature, on_step=False, on_epoch=True, prog_bar=True)

    def forward(self, *args, **kwargs):
        # This is called for prediction.
        # It passes the full batch dictionary to the model's forward method.
        return self.model(*args, **kwargs)
        
    def training_step(self, batch, batch_idx: int):
        # The model.loss method will select the inputs it needs from the batch dict.
        train_loss = self.model.loss(**batch)[0] 
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        # The model.loss method will select the inputs it needs from the batch dict.
        val_loss = self.model.loss(**batch)[0]
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        monitor = "val_loss" if has_validation_loop(self.trainer) else "train_loss"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.patience,
                ),
                "monitor": monitor,
            },
        }
    def forward_for_evaluation(self, *args, **kwargs):
        """
        Delegates the call to the underlying TimePrismNetwork's
        dedicated evaluation method. This is called by the custom
        evaluation loop to get all hypotheses and their probabilities.
        """
        # This acts as a bridge, passing the call to the actual network model.
        return self.model.forward_for_evaluation(*args, **kwargs)