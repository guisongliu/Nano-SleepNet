import torch
import torch.nn as nn
import torch.nn.functional as F


def align_time(x, target_length):
    if x.size(-1) > target_length:
        return x[..., :target_length]
    if x.size(-1) < target_length:
        return F.pad(x, (0, target_length - x.size(-1)))
    return x


class SamePadConv1d(nn.Module):
    """Conv1d with SAME padding. The paper does not state an explicit padding rule."""

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


class DepthwiseSeparableConv1d(nn.Module):
    """dw Conv -> BN -> GELU -> pw Conv -> BN -> GELU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activate_last=True):
        super().__init__()
        self.dw = SamePadConv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = nn.GELU()
        self.pw = SamePadConv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU() if activate_last else nn.Identity()

    def forward(self, x):
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class InvertedBottleneck1D(nn.Module):
    """
    LWSleepNet bottleneck block from Fig. 2.

    Structure:
      pw Conv(inp, hidden) -> BN -> GELU
      dw Conv(ks) -> BN -> GELU
      pw Conv(hidden, out) -> BN
      shortcut add when shape allows; otherwise a 1x1 projection is used.

    The paper states dwConv kernel size in all bottleneck blocks is 9.
    """

    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=9, dropout=0.0):
        super().__init__()
        hidden_channels = hidden_channels or in_channels * 4
        self.pw1 = SamePadConv1d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dw = SamePadConv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            groups=hidden_channels,
            bias=True,
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.pw2 = SamePadConv1d(hidden_channels, out_channels, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else SamePadConv1d(in_channels, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.pw1(x)))
        out = self.act(self.bn2(self.dw(out)))
        out = self.dropout(out)
        out = self.bn3(self.pw2(out))
        identity = align_time(identity, out.size(-1))
        return self.act(out + identity)


class LightMRCNN(nn.Module):
    """
    Depthwise separable multi-resolution CNN in Fig. 3.

    Paper-confirmed settings:
      two branches with kernels 5 and 50;
      branch input depth 16;
      branch concat -> dw Conv(3) + Dropout(0.5) + Bottleneck(32,48).

    A small 1x1 stem maps raw single-channel EEG to depth 16 because the paper
    figure starts this module at "Input(depth=16)".
    """

    def __init__(self, in_channels=1, stem_channels=16, mid_channels=32, out_channels=48):
        super().__init__()
        branch_channels = mid_channels // 2
        self.stem = nn.Sequential(
            SamePadConv1d(in_channels, stem_channels, kernel_size=1, bias=True),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        self.small = DepthwiseSeparableConv1d(stem_channels, branch_channels, kernel_size=5)
        self.large = DepthwiseSeparableConv1d(stem_channels, branch_channels, kernel_size=50)
        self.dw3 = nn.Sequential(
            SamePadConv1d(mid_channels, mid_channels, kernel_size=3, groups=mid_channels, bias=True),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(0.5)
        self.bottleneck = InvertedBottleneck1D(
            mid_channels,
            out_channels,
            hidden_channels=mid_channels * 4,
            kernel_size=9,
        )

    def forward(self, x):
        x = self.stem(x)
        x_small = self.small(x)
        x_large = self.large(x)
        x = torch.cat([x_small, align_time(x_large, x_small.size(-1))], dim=1)
        x = self.dw3(x)
        x = self.dropout(x)
        return self.bottleneck(x)


class PatchMHA(nn.Module):
    """
    TFE MHA over non-overlapping temporal patches.

    The paper uses patch length 25, 8 heads, and repeats this module 3 times.
    """

    def __init__(self, channels, patch_length=25, num_heads=8):
        super().__init__()
        self.channels = channels
        self.patch_length = patch_length
        self.norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        batch_size, channels, length = x.shape
        remainder = length % self.patch_length
        if remainder:
            x = F.pad(x, (0, self.patch_length - remainder))
            length = x.size(-1)

        patches = x.view(batch_size, channels, length // self.patch_length, self.patch_length)
        tokens = patches.mean(dim=-1).transpose(1, 2).contiguous()
        tokens = self.norm(tokens)
        attended, _ = self.mha(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attended
        expanded = tokens.transpose(1, 2).repeat_interleave(self.patch_length, dim=-1)
        return expanded[..., :length]


class TFEBlock(nn.Module):
    """
    Temporal Feature Extraction block from Fig. 1 and Fig. 4.

    Paper settings:
      Bottleneck(48,64) -> MHA block repeated 3 times -> Bottleneck(64,80)
      patch length = 25, heads = 8, bottleneck dw kernel = 9.
    """

    def __init__(self, in_channels=48, attn_channels=64, out_channels=80, patch_length=25, heads=8, repeats=3):
        super().__init__()
        self.pre = InvertedBottleneck1D(
            in_channels,
            attn_channels,
            hidden_channels=in_channels * 4,
            kernel_size=9,
        )
        self.attn = nn.Sequential(*[PatchMHA(attn_channels, patch_length, heads) for _ in range(repeats)])
        self.post = InvertedBottleneck1D(
            attn_channels,
            out_channels,
            hidden_channels=attn_channels * 4,
            kernel_size=9,
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.attn(x)
        x = self.post(x)
        return x


class LWSleepNet(nn.Module):
    """
    PyTorch reproduction of LWSleepNet.

    Source checked against:
      Yang et al., "LWSleepNet: A lightweight attention-based deep learning
      model for sleep staging with single-channel EEG", Digital Health, 2023.

    Confirmed paper settings:
      - 30 s single-channel EEG, 100 Hz, input length 3000.
      - Light-MRCNN with small/large kernels 5 and 50.
      - Bottleneck blocks use dwConv kernel size 9.
      - TFE cuts patches of length 25.
      - MHA heads = 8, TFE repeat count = 3.
      - Output block: Conv pw(80,32) -> AveragePool1d -> FC -> Softmax.

    The paper reports 180 K parameters and 55.3 MFLOPs. It does not publish
    code-level details for padding, stem projection, or projection shortcuts;
    these are implemented minimally so the Fig. 1-Fig. 4 tensor flow is
    executable in PyTorch.
    """

    def __init__(self, num_classes=5, input_channels=1, return_probs=False):
        super().__init__()
        self.return_probs = return_probs
        self.representation = LightMRCNN(input_channels, stem_channels=16, mid_channels=32, out_channels=48)
        self.tfe = TFEBlock(in_channels=48, attn_channels=64, out_channels=80, patch_length=25, heads=8, repeats=3)
        self.output_conv = nn.Sequential(
            SamePadConv1d(80, 32, kernel_size=1, bias=True),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

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
        x = self.representation(x)
        x = self.tfe(x)
        x = self.output_conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        if self.return_probs:
            x = F.softmax(x, dim=-1)
        if batch_size is not None:
            x = x.view(batch_size, windows, -1)
        return x


def count_paper_parameters(model):
    """Count Conv/Linear/MHA projection parameters; exclude normalization affine."""
    total = 0
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            total += module.in_proj_weight.numel()
            if module.in_proj_bias is not None:
                total += module.in_proj_bias.numel()
        elif isinstance(module, (nn.Conv1d, nn.Linear)):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()
    return total


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class cnn_new(LWSleepNet):
    pass


if __name__ == "__main__":
    model = LWSleepNet()
    x = torch.randn(2, 1, 3000)
    seq_x = torch.randn(2, 10, 1, 3000)
    print(model)
    print("single epoch output:", tuple(model(x).shape))
    print("sequence output:", tuple(model(seq_x).shape))
    print("paper-like parameter count:", count_paper_parameters(model))
    print("trainable parameters:", count_trainable_parameters(model))
