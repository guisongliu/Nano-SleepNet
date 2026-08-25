import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    if groups == 1:
        return x
    batch_size, channels, length = x.size()
    if channels % groups != 0:
        raise ValueError(f"channels={channels} must be divisible by groups={groups}")
    x = x.view(batch_size, groups, channels // groups, length)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, length)


class SamePadConv1d(nn.Module):
    """Conv1d with SAME padding for stride/dilation settings used in the paper."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        in_len = x.size(-1)
        out_len = (in_len + self.stride - 1) // self.stride
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        pad = max((out_len - 1) * self.stride + effective_kernel - in_len, 0)
        if pad:
            x = F.pad(x, (pad // 2, pad - pad // 2))
        return self.conv(x)


class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super().__init__()
        self.conv = SamePadConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv1d(nn.Module):
    """
    Depthwise separable Conv1D as shown in Fig. 2.

    The box label DSConv1D(Cout, k, s) is implemented as:
      depthwise Conv1D(Cout, k, s), groups=Cin
      pointwise Conv1D(Cout, 1, 1)
      BN + GELU
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.depthwise = SamePadConv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            bias=True,
        )
        self.dw_bn = nn.BatchNorm1d(in_channels)
        self.pointwise = SamePadConv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.pw_bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.dw_bn(self.depthwise(x)))
        x = self.act(self.pw_bn(self.pointwise(x)))
        return x


class DBCNN(nn.Module):
    """
    Dual-Branch CNN from EfficientSleepNet Fig. 2.

    Confirmed settings:
      Conv1D(64, 8, 2)
      DSConv1D(64, 8, 1)
      wide branch:   DSConv1D(64, 200, 25) -> MaxPooling(8, 8)
      narrow branch: DSConv1D(64, 25, 3)   -> MaxPooling(4, 4)

    The paper does not explicitly state whether branch fusion is channel- or
    time-axis concatenation. Their branch strides/pools produce very different
    temporal lengths, so this implementation concatenates along the time axis,
    preserving 64 channels for the following EDFE Conv1D(64, 8, 1).
    """

    def __init__(self, in_channels=1, channels=64):
        super().__init__()
        self.stem = ConvBNGELU(in_channels, channels, kernel_size=8, stride=2)
        self.pre = DSConv1d(channels, channels, kernel_size=8, stride=1)
        self.wide = nn.Sequential(
            DSConv1d(channels, channels, kernel_size=200, stride=25),
            nn.MaxPool1d(kernel_size=8, stride=8),
        )
        self.narrow = nn.Sequential(
            DSConv1d(channels, channels, kernel_size=25, stride=3),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

    def forward(self, x):
        x = self.pre(self.stem(x))
        wide = self.wide(x)
        narrow = self.narrow(x)
        return torch.cat([wide, narrow], dim=-1)


class EDFE(nn.Module):
    """
    Efficient Deep Feature Extraction from Fig. 2 and Sec. II-C.

    Four grouped dilated Conv1D layers:
      Conv1D(64, 8, 1), groups=4, dilation=1
      Conv1D(64, 8, 1), groups=4, dilation=2
      Conv1D(64, 8, 1), groups=4, dilation=4
      Conv1D(64, 8, 1), groups=4, dilation=8
    Each convolution is followed by GELU and Channel Shuffle, then
    MaxPooling(4, 4).
    """

    def __init__(self, channels=64, groups=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConvBNGELU(channels, channels, kernel_size=8, stride=1, dilation=d, groups=groups)
                for d in (1, 2, 4, 8)
            ]
        )
        self.groups = groups
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = channel_shuffle(x, self.groups)
        return self.pool(x)


class AFE(nn.Module):
    """
    Adaptive Feature Enhancement from Sec. II-D.

    The paper explicitly states:
      F_new(c) = F_avg(c) + F_max(c)
      F_new is processed by two 1D convolutional layers with ReLU;
      Sigmoid gives channel attention weights;
      original feature map is recalibrated by element-wise multiplication.

    The two Conv1D kernel sizes are not specified in the paper. Following the
    ECA-style channel-attention design cited by the authors, kernel_size=3 is
    used by default over the channel descriptor.
    """

    def __init__(self, channels=64,kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=True)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        maxv = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        descriptor = (avg + maxv).unsqueeze(1)
        weights = self.relu(self.conv1(descriptor))
        weights = self.sigmoid(self.conv2(weights)).squeeze(1).unsqueeze(-1)
        return x * weights


class EfficientSleepNetEncoder(nn.Module):
    """
    Encoder-only reproduction of EfficientSleepNet, with BiLSTM removed.

    Source checked against:
      Wang et al., "EfficientSleepNet: A Novel Lightweight End-to-End Model for
      Automated Sleep Staging on Single-Channel EEG", ICASSP 2025,
      DOI: 10.1109/ICASSP49660.2025.10889937.

    Encoded part implemented:
      DBCNN -> EDFE -> AFE

    Omitted by request:
      Bi-LSTM and FC sequence-classification layers.

    Input:
      [B, T], [B, 1, T], or [B, W, 1, T], where T=3000 for 30 s EEG at 100 Hz.
    Output:
      feature map [B, 64, L] or [B, W, 64, L].
    """

    def __init__(self, input_channels=1, channels=64, afe_kernel_size=3):
        super().__init__()
        self.dbcnn = DBCNN(input_channels, channels)
        self.edfe = EDFE(channels, groups=4)
        self.afe = AFE(channels, kernel_size=afe_kernel_size)

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
        x = self.dbcnn(x)
        x = self.edfe(x)
        x = self.afe(x)
        if batch_size is not None:
            x = x.view(batch_size, windows, x.size(1), x.size(2))
        return x


class EfficientSleepNetNoBiLSTM(nn.Module):
    """
    Practical no-BiLSTM classifier wrapper around the encoder.

    This is not the paper's sequence classifier. It is provided so existing
    training scripts can import cnn_new and receive [B, 5] logits directly.
    """

    def __init__(self, num_classes=5, input_channels=1, channels=64, afe_kernel_size=3):
        super().__init__()
        self.encoder = EfficientSleepNetEncoder(input_channels, channels, afe_kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        if features.dim() == 4:
            batch_size, windows, channels, length = features.shape
            features = features.reshape(batch_size * windows, channels, length)
            logits = self.fc(self.pool(features).squeeze(-1))
            return logits.view(batch_size, windows, -1)
        return self.fc(self.pool(features).squeeze(-1))


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_conv_linear_parameters(model):
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()
    return total


class cnn_new(EfficientSleepNetNoBiLSTM):
    pass


if __name__ == "__main__":
    encoder = EfficientSleepNetEncoder()
    classifier = EfficientSleepNetNoBiLSTM()
    x = torch.randn(2, 1, 3000)
    seq_x = torch.randn(2, 10, 1, 3000)

    y = encoder(x)
    seq_y = encoder(seq_x)
    logits = classifier(x)
    seq_logits = classifier(seq_x)

    print(encoder)
    print("encoder output:", tuple(y.shape))
    print("sequence encoder output:", tuple(seq_y.shape))
    print("no-BiLSTM logits:", tuple(logits.shape))
    print("sequence no-BiLSTM logits:", tuple(seq_logits.shape))
    print("encoder conv/linear parameters:", count_conv_linear_parameters(encoder))
    print("encoder trainable parameters:", count_trainable_parameters(encoder))
    print("classifier trainable parameters:", count_trainable_parameters(classifier))
