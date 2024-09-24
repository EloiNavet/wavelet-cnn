# downsample.py
from torch.nn import Module
import torch

from DWT_IDWT_layer import DWT_2D_Low, DWT_2D


class Downsample(Module):
    """
    Downsample module using 2D Discrete Wavelet Transform (DWT).

    Args:
        wavename (str, optional): Name of the wavelet. Defaults to "haar".
    """

    def __init__(self, wavename: str = "haar") -> None:
        super(Downsample, self).__init__()
        self.dwt = DWT_2D_Low(wavename=wavename)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsample module.

        Args:
            input (torch.Tensor): Input tensor to be downsampled of shape (B, C, H, W).

        Returns:
            torch.Tensor: Downsampled tensor of shape (B, C, H // 2, W // 2).
        """
        input_ll = self.dwt(input)
        return input_ll


class DownsampleConcat(Module):
    """
    Downsample module using 2D Discrete Wavelet Transform (DWT) and concatenation.

    Args:
        wavename (str, optional): Name of the wavelet. Defaults to "haar".
    """

    def __init__(self, wavename: str = "haar") -> None:
        super(DownsampleConcat, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsample module.

        Args:
            input (torch.Tensor): Input tensor to be downsampled of shape (B, C, H, W).

        Returns:
            torch.Tensor: Downsampled and concatenated tensor of shape (B, 2 * C, H // 2, W // 2).
        """
        input_ll, input_lh, input_hl, input_hh = self.dwt(input)
        return torch.cat([input_ll, input_lh + input_hl + input_hh], dim=1)


class DownsampleSum(Module):
    """
    Downsample module using 2D Discrete Wavelet Transform (DWT) and summation.

    Args:
        wavename (str, optional): Name of the wavelet. Defaults to "haar".
    """

    def __init__(self, wavename: str = "haar") -> None:
        super(DownsampleSum, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsample module.

        Args:
            input (torch.Tensor): Input tensor to be downsampled of shape (B, C, H, W).

        Returns:
            torch.Tensor: Downsampled and summed tensor of shape (B, C, H // 2, W // 2).
        """
        input_ll, input_lh, input_hl, input_hh = self.dwt(input)
        return torch.sum(input_ll + input_lh + input_hl + input_hh, dim=[2, 3])
