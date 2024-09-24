# vgg.py
import torch
from torch.nn import (
    Module,
    ReLU,
    Conv2d,
    AdaptiveAvgPool2d,
    Linear,
    Dropout,
    Sequential,
    BatchNorm2d,
)

from downsample import Downsample


class VGG(Module):
    def __init__(self, num_classes: int = 1000, wavename: str = "haar"):
        """
        VGG model with wavelet downsampling. The architecture is based on the original VGG paper (https://arxiv.org/abs/1409.1556).

        Args:
            num_classes (int, optional): Number of classes. Defaults to 1000.
            wavename (str, optional): Name of the wavelet. Defaults to "haar".
        """
        super(VGG, self).__init__()

        self.features = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Downsample(wavename=wavename),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Downsample(wavename=wavename),
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Downsample(wavename=wavename),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Downsample(wavename=wavename),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            ReLU(inplace=True),
            Downsample(wavename=wavename),
        )

        self.avgpool = AdaptiveAvgPool2d((7, 7))

        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            Linear(4096, 4096, bias=True),
            ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            Linear(4096, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VGG model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.classifier(x)
        return x


def vgg(**kwargs) -> VGG:
    """
    Create an instance of the VGG model.

    Args:
        **kwargs: Keyword arguments to pass to the VGG constructor.

    Returns:
        VGG: An instance of the VGG model.
    """
    model = VGG(**kwargs)
    return model
