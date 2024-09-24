# resnet.py
import torch
from torch.nn import (
    Module,
    ReLU,
    Conv2d,
    AdaptiveAvgPool2d,
    Linear,
    Sequential,
    BatchNorm2d,
)

from downsample import Downsample

# Since this is more complicated than VGG and AlexNET, I adapted the code from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

NUM_CHANNELS = 4


class Bottleneck(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Module = None,
        wavename: str = "haar",
    ):
        """
        Bottleneck block for ResNet. The block consists of 3 convolutional layers.

        Args:
            in_channels (int): Number of input channels to the block.
            out_channels (int): Number of output channels from the block.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.
            downsample (Module, optional): Downsample module. Defaults to None.
            wavename (str, optional): Name of the wavelet. See downsample.py for more details. Defaults to "haar".
        """
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = BatchNorm2d(out_channels)
        if stride == 1:
            self.conv3 = Conv2d(
                out_channels, out_channels * NUM_CHANNELS, kernel_size=1, bias=False
            )
        else:
            self.conv3 = Sequential(
                Downsample(wavename=wavename),
                Conv2d(
                    out_channels, out_channels * NUM_CHANNELS, kernel_size=1, bias=False
                ),
            )
        self.bn3 = BatchNorm2d(out_channels * NUM_CHANNELS)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the bottleneck block.

        Args:
            x (torch.Tensor): Input tensor to the block of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor from the block of shape (B, C', H', W').
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(x)
        out = self.relu(out)

        return out


class ResNet(Module):
    def __init__(
        self,
        block: Module,
        layers: list[int],
        num_classes: int = 1000,
        wavename: str = "haar",
    ):
        """
        ResNet model with the specified block and layers.

        Args:
            block (Module): Block to use in the ResNet model.
            layers (list[int]): List of integers specifying the number of blocks in each layer.
            num_classes (int, optional): Number of classes. Defaults to 1000.
            wavename (str, optional): Name of the wavelet. See downsample.py for more details. Defaults to "haar".
        """
        super(ResNet, self).__init__()
        out_channels = [64 * 2**i for i in range(4)]
        self.in_channels = out_channels[0]

        self.conv1 = Conv2d(
            3, out_channels[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = BatchNorm2d(out_channels[0])
        self.relu = ReLU(inplace=True)

        self.maxpool = Sequential(Downsample(wavename=wavename))

        self.layer1 = self._make_layer(block, out_channels[0], layers[0])
        self.layer2 = self._make_layer(
            block, out_channels[1], layers[1], stride=2, wavename=wavename
        )
        self.layer3 = self._make_layer(
            block, out_channels[2], layers[2], stride=2, wavename=wavename
        )
        self.layer4 = self._make_layer(
            block, out_channels[3], layers[3], stride=2, wavename=wavename
        )
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(out_channels[3] * NUM_CHANNELS, num_classes)

    def _make_layer(
        self,
        block,
        out_channels,
        blocks,
        stride=1,
        wavename="haar",
    ):
        """
        Make a layer of blocks for the ResNet model.

        Args:
            block (Module): Block to use in the layer.
            out_channels (int): Number of output channels from the block.
            blocks (int): Number of blocks in the layer.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.
            wavename (str, optional): Name of the wavelet. See downsample.py for more details. Defaults to "haar".

        Returns:
            Sequential: Layer of blocks.
        """
        downsample = (
            [
                Downsample(wavename=wavename),
            ]
            if stride == 2
            else []
        )
        downsample += [
            Conv2d(
                self.in_channels,
                out_channels * NUM_CHANNELS,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            BatchNorm2d(out_channels * NUM_CHANNELS),
        ]
        downsample = Sequential(*downsample)

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample, wavename=wavename)
        )
        self.in_channels = out_channels * NUM_CHANNELS
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, wavename=wavename))

        return Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor to the model of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor from the model of shape (B, num_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet101(wavename: str = "haar", **kwargs) -> ResNet:
    """
    Create an instance of the ResNet-101 model.

    Args:
        wavename (str, optional): Name of the wavelet. See downsample.py for more details. Defaults to "haar".

    Returns:
        ResNet: Instance of the ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], wavename=wavename, **kwargs)
    return model
