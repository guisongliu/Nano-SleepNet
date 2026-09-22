"""NanoSleepNet-TCN: lightweight sequence-to-sequence sleep staging model.

The model contains two parts:

1. LightShuffleEEG100HzExtractor extracts features from every 30-second EEG
   epoch. Its output is [B, 64, 32] for one epoch.
2. LightweightDS-TCN models the temporal relationship between consecutive
   epochs. It receives [B, W, 64] and returns [B, W, 64].

Supported input shapes:
    [B, T]          -> one epoch per sample, output [B, num_classes]
    [B, C, T]       -> one epoch per sample, output [B, num_classes]
    [B, W, C, T]    -> W consecutive epochs, output [B, W, num_classes]

For 100 Hz, 30-second EEG, T is normally 3000.
"""

import math
from typing import Iterable, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv1d(nn.Module):
    """1-D convolution with TensorFlow-like SAME output length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        input_length = x.size(-1)
        output_length = math.ceil(input_length / self.stride)
        padding = max(
            (output_length - 1) * self.stride + self.kernel_size - input_length,
            0,
        )
        if padding:
            left = padding // 2
            right = padding - left
            x = F.pad(x, (left, right))
        return self.conv(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be positive")
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, length = x.shape
        if channels % self.groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by groups={self.groups}"
            )
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, length)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, channels, length)


class ConvBNGELU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        shuffle_groups: int = 0,
    ):
        super().__init__()
        self.conv = SamePadConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.shuffle = (
            ChannelShuffle(shuffle_groups)
            if shuffle_groups > 1
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.shuffle(self.act(self.bn(self.conv(x))))


class ShuffleResidualBlock(nn.Module):
    """Grouped temporal convolutions with channel mixing and a residual path."""

    def __init__(self, channels: int = 32, groups: int = 4):
        super().__init__()
        self.conv1 = ConvBNGELU(
            channels,
            channels,
            kernel_size=15,
            stride=1,
            groups=groups,
            shuffle_groups=groups,
        )
        self.conv2 = SamePadConv1d(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class TinyECA(nn.Module):
    """Efficient channel attention with only a few trainable parameters."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pool along the within-epoch time axis and generate one weight per
        # feature channel.
        weights = x.mean(dim=-1, keepdim=False).unsqueeze(1)
        weights = torch.sigmoid(self.conv(weights))
        weights = weights.squeeze(1).unsqueeze(-1)
        return x * weights


class LightShuffleEEG100HzExtractor(nn.Module):
    """NanoSleepNet local feature extractor for one EEG epoch.

    For input length 3000 the temporal sizes are approximately:

        3000 -> 500 -> 500 -> 125 -> 32

    The actual output is [B, 64, 32]. The output channel count is 64 because
    the original lightweight NanoSleepNet implementation does not activate
    the optional 64 -> 128 expansion layer.
    """

    output_channels = 64

    def __init__(self, input_channels: int = 1, use_eca: bool = True):
        super().__init__()
        self.stem = ConvBNGELU(
            input_channels,
            32,
            kernel_size=15,
            stride=6,
        )
        self.residual = ShuffleResidualBlock(channels=32, groups=4)
        self.downsample = ConvBNGELU(
            32,
            64,
            kernel_size=9,
            stride=4,
            groups=8,
            shuffle_groups=8,
        )
        self.fixed_pool = nn.AvgPool1d(
            kernel_size=4,
            stride=4,
            ceil_mode=True,
        )
        self.eca = TinyECA(kernel_size=5) if use_eca else nn.Identity()

    @staticmethod
    def _normalize_input(x: Tensor) -> Tuple[Tensor, int, int]:
        if x.dim() == 2:
            # [B, T] -> [B, 1, T], one epoch per sample.
            return x.unsqueeze(1), x.size(0), 1
        if x.dim() == 3:
            # [B, C, T], one epoch per sample.
            return x, x.size(0), 1
        if x.dim() == 4:
            # [B, W, C, T] -> [B*W, C, T].
            batch_size, windows, channels, length = x.shape
            return x.reshape(batch_size * windows, channels, length), batch_size, windows
        raise ValueError(
            "Expected [B,T], [B,C,T], or [B,W,C,T], "
            f"but received {tuple(x.shape)}"
        )

    def forward(self, x: Tensor) -> Tensor:
        x, batch_size, windows = self._normalize_input(x)
        x = self.stem(x)
        x = self.residual(x)
        x = self.downsample(x)
        x = self.fixed_pool(x)
        x = self.eca(x)

        if windows > 1:
            x = x.view(batch_size, windows, x.size(1), x.size(2))
        return x


class DepthwiseSeparableTemporalBlock(nn.Module):
    """Lightweight temporal block used by the sequence model.

    It first reduces channels, then performs depthwise temporal filtering and
    pointwise channel mixing. The dilation blocks are applied serially, so the
    effective temporal receptive field grows progressively.
    """

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size
        padding = dilation * (kernel_size - 1)
        symmetric_padding = padding // 2

        self.depthwise = nn.Conv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0 if causal else symmetric_padding,
            dilation=dilation,
            groups=bottleneck_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm1d(bottleneck_channels)
        self.pointwise = nn.Conv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_bn = nn.BatchNorm1d(bottleneck_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.causal:
            left_padding = self.dilation * (self.kernel_size - 1)
            x = F.pad(x, (left_padding, 0))
        x = self.depthwise(x)
        x = self.act(self.depthwise_bn(x))
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.act(x)
        return self.dropout(x)


class LightweightDSTCN(nn.Module):
    """Serial bottlenecked depthwise-separable TCN.

    Input and output shape: [B, 64, W].
    """

    def __init__(
        self,
        channels: int = 64,
        bottleneck_channels: int = 32,
        kernel_size: int = 3,
        dilations: Iterable[int] = (1, 2, 4),
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels <= 0 or bottleneck_channels <= 0:
            raise ValueError("channels and bottleneck_channels must be positive")

        self.reduce = nn.Sequential(
            nn.Conv1d(channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                DepthwiseSeparableTemporalBlock(
                    channels=channels,
                    bottleneck_channels=bottleneck_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    causal=causal,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.expand = nn.Sequential(
            nn.Conv1d(bottleneck_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.reduce(x)
        x = self.blocks(x)
        x = self.expand(x)
        return self.act(x + residual)


class NanoSleepNetTCN(nn.Module):
    """NanoSleepNet with sequence-to-sequence temporal modeling.

    Input [B, W, 1, 3000] produces one prediction for each of the W epochs:
    [B, W, num_classes].
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 1,
        temporal_bottleneck: int = 32,
        temporal_kernel_size: int = 3,
        temporal_dilations: Iterable[int] = (1, 2, 4),
        causal: bool = False,
        dropout: float = 0.1,
        use_eca: bool = True,
        return_probs: bool = False,
    ):
        super().__init__()
        self.return_probs = return_probs
        self.extractor = LightShuffleEEG100HzExtractor(
            input_channels=input_channels,
            use_eca=use_eca,
        )
        self.temporal = LightweightDSTCN(
            channels=64,
            bottleneck_channels=temporal_bottleneck,
            kernel_size=temporal_kernel_size,
            dilations=temporal_dilations,
            causal=causal,
            dropout=dropout,
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.extractor(x)

        if features.dim() == 3:
            # [B, 64, 32] -> [B, 1, 64].
            features = features.mean(dim=-1).unsqueeze(1)
        elif features.dim() == 4:
            # [B, W, 64, 32] -> [B, W, 64].
            features = features.mean(dim=-1)
        else:
            raise RuntimeError(f"Unexpected extractor output: {features.shape}")

        # TCN uses [batch, channels, sequence].
        sequence_features = features.transpose(1, 2).contiguous()
        sequence_features = self.temporal(sequence_features)
        sequence_features = sequence_features.transpose(1, 2).contiguous()
        logits = self.classifier(sequence_features)

        if self.return_probs:
            logits = torch.softmax(logits, dim=-1)

        # Preserve the convenient single-epoch output convention.
        if x.dim() in (2, 3):
            return logits[:, 0, :]
        return logits


# Compatibility aliases for existing training scripts and registries.
NanoSleepNet_TCN = NanoSleepNetTCN
cnn_new = NanoSleepNetTCN


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


if __name__ == "__main__":
    torch.manual_seed(42)
    model = NanoSleepNetTCN(num_classes=5)
    model.eval()

    one_epoch = torch.randn(2, 1, 3000)
    sequence = torch.randn(2, 10, 1, 3000)

    with torch.no_grad():
        one_epoch_logits = model(one_epoch)
        sequence_logits = model(sequence)

    total, trainable = count_parameters(model)
    print("one epoch input :", tuple(one_epoch.shape))
    print("one epoch output:", tuple(one_epoch_logits.shape))
    print("sequence input  :", tuple(sequence.shape))
    print("sequence output :", tuple(sequence_logits.shape))
    print(f"parameters      : {total:,}")
    print(f"trainable       : {trainable:,}")

