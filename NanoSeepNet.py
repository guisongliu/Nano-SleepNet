import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv1d(nn.Module):
    """Conv1d with dynamic SAME padding for stride > 1."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        in_len = x.size(-1)
        out_len = math.ceil(in_len / self.stride)
        pad = max((out_len - 1) * self.stride + self.kernel_size - in_len, 0)
        left = pad // 2
        right = pad - left
        if pad > 0:
            x = F.pad(x, (left, right))
        return self.conv(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, length = x.shape
        if channels % self.groups != 0:
            raise ValueError(f"channels={channels} must be divisible by groups={self.groups}")
        x = x.view(batch_size, self.groups, channels // self.groups, length)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, channels, length)


class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, shuffle_groups=None):
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
        self.shuffle = ChannelShuffle(shuffle_groups) if shuffle_groups else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.shuffle(x)


class ShuffleResidualBlock(nn.Module):
    """Grouped temporal convolution + channel shuffle + residual enhancement."""

    def __init__(self, channels=32, groups=4):
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

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class TinyECA(nn.Module):
    """Efficient channel attention with only kernel_size trainable weights."""

    def __init__(self, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        weight = x.mean(dim=-1, keepdim=False).unsqueeze(1)
        weight = torch.sigmoid(self.conv(weight)).squeeze(1).unsqueeze(-1)
        return x * weight


class LightShuffleEEG100HzExtractor(nn.Module):
    """
    Lightweight EEG feature extractor for 100 Hz, 30 s epochs.

    Input:
        [B, T], [B, 1, 3000], or [B, W, 1, 3000]

    Output:
        [B, 128, 188] or [B, W, 128, 188]
    """

    def __init__(self, input_channels=1, output_channels=128, use_eca=True):
        super().__init__()
        if output_channels != 128:
            raise ValueError("This extractor is fixed to output_channels=128.")

        self.stem = ConvBNGELU(input_channels, 32, kernel_size=15, stride=6)
        self.residual = ShuffleResidualBlock(channels=32, groups=4)#   设定为8组呢？
        self.downsample = ConvBNGELU(
            32,
            64,
            kernel_size=9,
            stride=4,
            groups=8,
            shuffle_groups=8,
        )
        # self.expand = ConvBNGELU(
        #     64,
        #     128,
        #     kernel_size=1,
        #     stride=1,
        #     groups=16,
        #     shuffle_groups=16,
        # )
        self.dropout = nn.Dropout(p=0.1)
        self.fixed_pool = nn.AvgPool1d(kernel_size=4, stride=4, ceil_mode=True)
        self.eca = TinyECA(kernel_size=5) if use_eca else nn.Identity()

    def _normalize_input(self, x):
        if x.dim() == 2:
            return x.unsqueeze(1), None, None
        if x.dim() == 3:
            return x, None, None
        if x.dim() == 4:
            batch_size, windows, channels, length = x.shape
            return x.reshape(batch_size * windows, channels, length), batch_size, windows
        raise ValueError(f"expected [B,T], [B,C,T], or [B,W,C,T], got {tuple(x.shape)}")

    def forward(self, x):
        x, batch_size, windows = self._normalize_input(x)

        x = self.stem(x)          # 3000 -> 500
        x = self.residual(x)      # 500 -> 500
        #x = self.dropout(x)

        x = self.downsample(x)    # 500 -> 125
        # x = self.expand(x)        # 125 -> 125
        #x = self.dropout(x)

        x = self.fixed_pool(x)    # 125 -> 32
        x = self.eca(x)           # [B, 128, 188]

        if batch_size is not None:
            x = x.view(batch_size, windows, x.size(1), x.size(2))
        return x


class DepthwiseSeparableDilatedBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(DepthwiseSeparableDilatedBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleDSFusion(nn.Module):
    """Serial lightweight replacement for the original dilated Conv1d stack.

    Tensor flow:
        [B, 128, T]
        -> 1x1 bottleneck, 128 -> 32
        -> depthwise-separable dilated block, dilation=1
        -> depthwise-separable dilated block, dilation=2
        -> depthwise-separable dilated block, dilation=4
        -> 1x1 expansion, 32 -> 128
        -> residual add
        -> [B, 128, T]
    """

    def __init__(self, channels=128, bottleneck_channels=32, dilations=(1, 2, 4), use_residual=True):
        super(MultiScaleDSFusion, self).__init__()
        self.use_residual = use_residual
        self.reduce = nn.Sequential(
            nn.Conv1d(
                channels,
                bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(bottleneck_channels),
            nn.GELU(),
        )
        self.serial_dilated = nn.Sequential(
            *[
                DepthwiseSeparableDilatedBlock(
                    bottleneck_channels,
                    kernel_size=3,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.expand = nn.Sequential(
            nn.Conv1d(
                bottleneck_channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        y = self.reduce(x)
        y = self.serial_dilated(y)
        y = self.expand(y)

        if self.use_residual:
            y = y + x

        return self.act(y)


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, width)

    return x


class TemporalClassifierHead(nn.Module):
    """Global temporal pooling classifier for epoch-level sleep staging."""

    def __init__(self, channels=128, num_classes=5, dropout=0.3, return_probs=False):
        super().__init__()
        self.return_probs = return_probs
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.fc(x)
        if self.return_probs:
            return torch.softmax(logits, dim=-1)
        return logits




class NanoSleepNet(nn.Module):
    """
    Trainable baseline:
        LightShuffleEEG100HzExtractor -> GAP -> FC
    """

    def __init__(self, num_classes=5, input_channels=1, dropout=0.0, return_probs=False):
        super().__init__()
        self.extractor = LightShuffleEEG100HzExtractor(input_channels=input_channels, output_channels=128)
        self.classifier = TemporalClassifierHead(
            channels=64,
            num_classes=num_classes,
            dropout=dropout,
            return_probs=return_probs,
        )

    def forward(self, x):
        features = self.extractor(x)
        if features.dim() == 4:
            batch_size, windows, channels, length = features.shape
            features = features.reshape(batch_size * windows, channels, length)
            logits = self.classifier(features)
            return logits.view(batch_size, windows, -1)
        return self.classifier(features)


class NanoSleepNet_TCN(nn.Module):
    """
    Trainable fusion model:
        LightShuffleEEG100HzExtractor -> MultiScaleDSFusion -> GAP -> FC
    """

    def __init__(
        self,
        num_classes=5,
        input_channels=1,
        bottleneck_channels=32,
        dilations=(1, 2, 4),
        dropout=0.5,
        return_probs=False,
    ):
        super().__init__()
        self.extractor = LightShuffleEEG100HzExtractor(input_channels=input_channels, output_channels=128)
        self.fusion = MultiScaleDSFusion(
            channels=128,
            bottleneck_channels=bottleneck_channels,
            dilations=dilations,
            use_residual=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = TemporalClassifierHead(
            channels=128,
            num_classes=num_classes,
            dropout=dropout,
            return_probs=return_probs,
        )

    def forward(self, x):
        features = self.extractor(x)
        x = self.dropout(x)
        if features.dim() == 4:
            batch_size, windows, channels, length = features.shape
            features = features.reshape(batch_size * windows, channels, length)
            features = self.fusion(features)
            logits = self.classifier(features)
            return logits.view(batch_size, windows, -1)

        features = self.fusion(features)
        return self.classifier(features)

