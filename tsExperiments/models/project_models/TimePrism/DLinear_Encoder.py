# tsExperiments/models/project_models/TimePrism/DLinear_Encoder.py

import torch
from torch import nn
from torch import Tensor
from typing import Tuple

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    This is a standard component from the DLinear model.
    """
    def __init__(self, kernel_size: int, stride: int):
        # super(moving_avg, self).__init__()
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # padding on the both ends of time series
        # front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block.
    This is a standard component from the DLinear model.
    """
    def __init__(self, kernel_size: int):
        # super(series_decomp, self).__init__()
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear_Encoder(nn.Module):
    """
    A lightweight, DLinear-inspired encoder.
    
    This encoder's sole function is to decompose the input time series into
    its seasonal and trend components. It contains no trainable parameters,
    acting as a pure pre-processing block that feeds into downstream networks.
    """
    def __init__(self, decomp_kernel_size: int = 25, **kwargs):
        # super(DLinear_Encoder, self).__init__()
        super().__init__()
        self.decomposition = series_decomp(decomp_kernel_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [bs x seq_len x nvars]
        seasonal_init, trend_init = self.decomposition(x)
        # The output shapes are the same as the input shape: [bs x seq_len x nvars]
        return seasonal_init, trend_init