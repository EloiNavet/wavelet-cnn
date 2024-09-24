# alexnet.py
import torch
from torch.nn import (
    Module,
    ReLU,
    Conv2d,
    AdaptiveAvgPool2d,
    Linear,
    Dropout,
    Sequential,
)

from downsample import Downsample


class AlexNet(Module):
    """
    AlexNet model with wavelet downsampling. The architecture is based on the original AlexNet paper (https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf).

    Args:
        num_classes (int, optional): Number of classes. Defaults to 1000.
        wavename (str, optional): Name of the wavelet. Defaults to "haar".
    """

    def __init__(self, num_classes: int = 1000, wavename: str = "haar") -> None:
        super(AlexNet, self).__init__()

        self.features = Sequential(
            Conv2d(3, 64, kernel_size=11, stride=2, padding=0),
            ReLU(inplace=True),
            Downsample(wavename=wavename),
            Downsample(wavename=wavename),
            Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU(inplace=True),
            Downsample(wavename=wavename),
            Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Downsample(wavename=wavename),
        )

        self.avgpool = AdaptiveAvgPool2d((6, 6))

        self.classifier = Sequential(
            Dropout(),
            Linear(256 * 6 * 6, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AlexNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(**kwargs) -> AlexNet:
    """
    Create an instance of the AlexNet model.

    Args:
        **kwargs: Keyword arguments to pass to the AlexNet constructor.

    Returns:
        AlexNet: An instance of the AlexNet model.
    """
    model = AlexNet(**kwargs)
    return model
