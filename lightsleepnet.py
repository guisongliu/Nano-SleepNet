import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    """Shuffle channels as used after each grouped 1D convolution."""
    if groups == 1:
        return x

    batch_size, channels, length = x.size()
    if channels % groups != 0:
        raise ValueError(f"channels={channels} must be divisible by groups={groups}")

    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, length)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, length)


class SamePadConv1d(nn.Module):
    """
    Conv1d with TensorFlow-style SAME padding.

    The paper gives kernel/stride/group settings in Fig. 1, but does not state
    the padding convention. SAME padding keeps residual additions well-defined
    for stride=1 and gives ceil(L / stride) for stride>1.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        in_len = x.size(-1)
        out_len = (in_len + self.stride - 1) // self.stride
        pad_needed = max((out_len - 1) * self.stride + self.kernel_size - in_len, 0)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        if pad_needed > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class LightSleepConvLayer(nn.Module):
    """Group Conv1d -> BatchNorm1d/AdaBN placeholder -> ReLU -> Channel Shuffle."""

    def __init__(self, in_channels, out_channels, kernel_size=16, stride=1, groups=1):
        super().__init__()
        self.groups = groups
        self.conv = SamePadConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = channel_shuffle(x, self.groups)
        return x


class IdentityShortcut1d(nn.Module):
    """
    Non-parametric shortcut for residual blocks with stride/channel changes.

    Fig. 1 shows residual addition in the second block although its first
    convolution uses stride=2 and Cout=128. A learned projection would add
    parameters and break the reported 43.08K count, so this follows the
    parameter-free shortcut used by small ResNet-style models.
    """

    def __init__(self, out_channels, stride=1):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x, target_length):
        if self.stride > 1:
            x = x[..., :: self.stride]

        if x.size(-1) > target_length:
            x = x[..., :target_length]
        elif x.size(-1) < target_length:
            x = F.pad(x, (0, target_length - x.size(-1)))

        if x.size(1) < self.out_channels:
            pad_channels = self.out_channels - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_channels))
        elif x.size(1) > self.out_channels:
            x = x[:, : self.out_channels, :]

        return x


class LightSleepResidualBlock(nn.Module):
    """Two grouped convolution layers with one residual addition."""

    def __init__(self, in_channels, out_channels, kernel_size=16, stride=1, groups=8):
        super().__init__()
        self.conv1 = LightSleepConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )
        self.conv2 = LightSleepConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=groups,
        )
        self.shortcut = IdentityShortcut1d(out_channels=out_channels, stride=stride)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.shortcut(identity, target_length=out.size(-1))
        return out + identity


class LightSleepNet(nn.Module):
    """
    PyTorch reproduction of LightSleepNet from arXiv:2401.13194v1.

    Architecture checked against Fig. 1 and Sec. II-A:
      - 30 s single-channel EEG epoch, Sleep-EDF uses 100 Hz by default.
      - Five 1D convolution layers with k=16.
      - Conv1: Cout=64, stride=2.
      - Residual block 1: two Conv layers, Cout=64, group=8.
      - Residual block 2: Conv layer Cout=128, stride=2, group=16;
        then Conv layer Cout=128, group=16.
      - Channel shuffle after each convolution layer.
      - Dropout p=0.5, global average pooling, Linear 128 -> 5.

    The paper reports 43.08K parameters. That matches Conv/Linear weights and
    biases only. PyTorch's trainable count is larger because BatchNorm affine
    parameters are trainable.
    """

    def __init__(
        self,
        num_classes=5,
        input_channels=1,
        dropout=0.5,
        pad_input_to_3008=True,
        return_probs=False,
    ):
        super().__init__()
        self.pad_input_to_3008 = pad_input_to_3008
        self.return_probs = return_probs

        self.stem = LightSleepConvLayer(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=16,
            stride=2,
            groups=1,
        )
        self.block1 = LightSleepResidualBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=16,
            stride=1,
            groups=8,
        )
        self.block2 = LightSleepResidualBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=16,
            stride=2,
            groups=16,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def _normalize_input_shape(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            pass
        elif x.dim() == 4:
            batch_size, windows, channels, length = x.shape
            x = x.reshape(batch_size * windows, channels, length)
            return x, batch_size, windows
        else:
            raise ValueError(
                "LightSleepNet expects input shape [B,T], [B,C,T], or [B,W,C,T], "
                f"but got {tuple(x.shape)}"
            )
        return x, None, None

    def _pad_epoch_length(self, x):
        if not self.pad_input_to_3008:
            return x
        if x.size(-1) == 3000:
            return F.pad(x, (4, 4))
        return x

    def forward(self, x):
        x, batch_size, windows = self._normalize_input_shape(x)
        x = self._pad_epoch_length(x)

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)

        if self.return_probs:
            x = F.softmax(x, dim=-1)

        if batch_size is not None:
            x = x.view(batch_size, windows, -1)
        return x


def count_paper_parameters(model):
    """
    Count parameters in the same way as Table I: Conv1d/Linear weights+biases.
    This excludes BatchNorm affine parameters, matching the reported 43.08K.
    """
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()
    return total


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Compatibility alias if an existing training script imports cnn_new.
class cnn_new(LightSleepNet):
    pass


if __name__ == "__main__":
    model = LightSleepNet()
    x = torch.randn(2, 1, 3000)
    y = model(x)
    seq_x = torch.randn(2, 10, 1, 3000)
    seq_y = model(seq_x)

    print(model)
    print("single epoch output:", tuple(y.shape))
    print("sequence output:", tuple(seq_y.shape))
    print("paper parameter count:", count_paper_parameters(model))
    print("trainable parameter count:", count_trainable_parameters(model))
