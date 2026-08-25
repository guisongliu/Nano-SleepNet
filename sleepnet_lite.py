import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups=2):
    if groups == 1:
        return x
    batch_size, channels, length = x.size()
    if channels % groups != 0:
        raise ValueError(f"channels={channels} must be divisible by groups={groups}")
    x = x.view(batch_size, groups, channels // groups, length)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, length)


def align_time(x, target_length):
    if x.size(-1) > target_length:
        return x[..., :target_length]
    if x.size(-1) < target_length:
        return F.pad(x, (0, target_length - x.size(-1)))
    return x


class SamePadConv1d(nn.Module):
    """Conv1d with SAME padding, needed because the paper does not specify padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
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
        out_len = (x.size(-1) + self.stride - 1) // self.stride
        pad = max((out_len - 1) * self.stride + self.kernel_size - x.size(-1), 0)
        if pad:
            x = F.pad(x, (pad // 2, pad - pad // 2))
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            SamePadConv1d(in_channels, out_channels, kernel_size, stride, groups, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DepthwisePointwiseBranch(nn.Module):
    """DW_Conv1d -> BN -> PW_Conv1d -> BN -> ReLU in Fig. 1(a)."""

    def __init__(self, channels, kernel_size, stride=2):
        super().__init__()
        self.dw = SamePadConv1d(channels, channels, kernel_size, stride=stride, groups=channels, bias=True)
        self.bn1 = nn.BatchNorm1d(channels)
        self.pw = SamePadConv1d(channels, channels, kernel_size=1, stride=1, groups=1, bias=True)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn1(self.dw(x))
        x = self.bn2(self.pw(x))
        return self.relu(x)


class MSCB(nn.Module):
    """
    Multi-Scale CNN Block from SleepNet-Lite Fig. 1(a).

    Paper settings for Sleep-EDF:
      Conv1d(k=8, out=64, s=2) -> BN -> ReLU
      MaxPooling(k=8, s=2, d=1)
      PW_Conv1d -> BN -> ReLU
      two DW-PW branches: k=5 and k=15, out=64, s=2
      Concat -> Channel Shuffle -> Channel Split
    """

    def __init__(self, in_channels=1, channels=64, first_kernel=8):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, channels, kernel_size=first_kernel, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=2, padding=3, dilation=1)
        self.pre_pw = ConvBNReLU(channels, channels, kernel_size=1, stride=1)
        self.branch_small = DepthwisePointwiseBranch(channels, kernel_size=5, stride=2)
        self.branch_large = DepthwisePointwiseBranch(channels, kernel_size=15, stride=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = self.pre_pw(x)
        x1 = self.branch_small(x)
        x2 = self.branch_large(x)
        if x1.size(-1) != x2.size(-1):
            target = min(x1.size(-1), x2.size(-1))
            x1 = align_time(x1, target)
            x2 = align_time(x2, target)
        x = torch.cat([x1, x2], dim=1)
        x = channel_shuffle(x, groups=2)
        return torch.chunk(x, chunks=2, dim=1)


class InvertedResidualFusion(nn.Module):
    """
    PW-DW-PW residual block from Fig. 1(a).

    One channel-split group is the shortcut; the other group passes through
    PW_Conv1d -> BN/ReLU -> DW_Conv1d(k=15,out=128,s=2) -> BN
    -> PW_Conv1d -> BN/ReLU, then both groups are concatenated.
    """

    def __init__(self, channels=64, hidden_channels=128, kernel_size=15):
        super().__init__()
        self.pw1 = ConvBNReLU(channels, hidden_channels, kernel_size=1, stride=1)
        self.dw = nn.Sequential(
            SamePadConv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=2,
                groups=hidden_channels,
                bias=True,
            ),
            nn.BatchNorm1d(hidden_channels),
        )
        self.pw2 = ConvBNReLU(hidden_channels, channels, kernel_size=1, stride=1)

    def forward(self, shortcut, x):
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        shortcut = F.avg_pool1d(shortcut, kernel_size=2, stride=2, ceil_mode=True)
        shortcut = align_time(shortcut, x.size(-1))
        return torch.cat([shortcut, x], dim=1)


class SleepNetLite(nn.Module):
    """
    PyTorch reproduction of SleepNet-Lite.

    Source checked against:
      Zhou et al., "SleepNet-Lite: A Novel Lightweight Convolutional Neural
      Network for Single-Channel EEG-Based Sleep Staging", IEEE Sensors
      Letters, 2023, Fig. 1 and Sec. II-A.

    Defaults reproduce the Sleep-EDF setting:
      30 s single-channel EEG, 100 Hz, input length 3000.
      The paper changes only the first kernel to 20 for MASS-SS3 at 256 Hz.
    """

    def __init__(self, num_classes=5, input_channels=1, first_kernel=8, final_channels=64, return_probs=False):
        super().__init__()
        self.return_probs = return_probs
        self.mscb = MSCB(input_channels, channels=64, first_kernel=first_kernel)
        self.residual = InvertedResidualFusion(channels=64, hidden_channels=128, kernel_size=15)
        self.classifier_conv = ConvBNReLU(128, final_channels, kernel_size=1, stride=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(final_channels, num_classes)

    def _normalize_input_shape(self, x):
        if x.dim() == 2:
            return x.unsqueeze(1), None, None
        if x.dim() == 3:
            return x, None, None
        if x.dim() == 4:
            batch_size, windows, channels, length = x.shape
            return x.reshape(batch_size * windows, channels, length), batch_size, windows
        raise ValueError(f"expected [B,T], [B,C,T], or [B,W,C,T], got {tuple(x.shape)}")

    def forward(self, x):
        x, batch_size, windows = self._normalize_input_shape(x)
        x_short, x_main = self.mscb(x)
        x = self.residual(x_short, x_main)
        x = self.classifier_conv(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        if self.return_probs:
            x = F.softmax(x, dim=-1)
        if batch_size is not None:
            x = x.view(batch_size, windows, -1)
        return x


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_paper_parameters(model):
    """Paper table count: Conv/Linear parameters, excluding BatchNorm affine."""
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()
    return total


class cnn_new(SleepNetLite):
    pass


if __name__ == "__main__":
    model = SleepNetLite()
    x = torch.randn(2, 1, 3000)
    seq_x = torch.randn(2, 10, 1, 3000)
    print(model)
    print("single epoch output:", tuple(model(x).shape))
    print("sequence output:", tuple(model(seq_x).shape))
    print("paper parameter count:", count_paper_parameters(model))
    print("trainable parameters:", count_trainable_parameters(model))
