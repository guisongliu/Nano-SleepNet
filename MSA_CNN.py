"""
Base classes for the MSA-CNN model (sgoerttler).
"""

import numpy as np
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Dropout(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(x)


def activation(x, config=None):
    if config is None:
        return F.relu(x)
    elif config['activation_function'] == 'relu':
        return F.relu(x)
    elif config['activation_function'] == 'gelu':
        return F.gelu(x)
    elif config['activation_function'] == 'leaky_relu':
        return F.leaky_relu(x)


class ScalingLayer(nn.Module):
    def __init__(self, num_features, bias_on=True):
        super(ScalingLayer, self).__init__()
        # Initialize the diagonal elements as a parameter with random weights
        self.weights = nn.Parameter(torch.ones(num_features))

        # Initialize the bias (if needed)
        self.bias_on = bias_on
        if self.bias_on:
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.swapaxes(x, 2, 3)
        x = x * self.weights
        if self.bias_on:
            x = x + self.bias
        return torch.swapaxes(x, 2, 3)


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Adjusted TransformerEncoderLayer to allow access to attention weights (sgoerttler)."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False, access_attention_weights=False, config=None):
        super(CustomTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)
        self.access_attention_weights = access_attention_weights
        self.config = config

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        if type(src) is tuple:
            return src
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)

                if self.access_attention_weights:
                    # Capture the query, key, and value before the attention is applied
                    query_w, key_w, value_w = self.self_attn.in_proj_weight.chunk(3)
                    query_b, key_b, value_b = self.self_attn.in_proj_bias.chunk(3)
                    query = src @ query_w.T + query_b
                    key = src @ key_w.T + key_b
                    qpp = query.reshape(src.shape[0], src.shape[1], self.config['num_heads'], self.config['embedding_dim'] // self.config['num_heads']).transpose(1, 2)
                    kpp = key.reshape(src.shape[0], src.shape[1], self.config['num_heads'], self.config['embedding_dim'] // self.config['num_heads']).transpose(1, 2)
                    attention_map = torch.nn.functional.softmax(qpp @ kpp.transpose(-1, -2) / np.sqrt(self.config['embedding_dim'] // self.config['num_heads']), dim=-1)
                    return src, attention_map
                else:
                    return torch._transformer_encoder_layer_fwd(
                        src,
                        self.self_attn.embed_dim,
                        self.self_attn.num_heads,
                        self.self_attn.in_proj_weight,
                        self.self_attn.in_proj_bias,
                        self.self_attn.out_proj.weight,
                        self.self_attn.out_proj.bias,
                        self.activation_relu_or_gelu == 2,
                        self.norm_first,
                        self.norm1.eps,
                        self.norm1.weight,
                        self.norm1.bias,
                        self.norm2.weight,
                        self.norm2.bias,
                        self.linear1.weight,
                        self.linear1.bias,
                        self.linear2.weight,
                        self.linear2.bias,
                        merged_mask,
                        mask_type,
                    )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=self.access_attention_weights, is_causal=is_causal)[0]
        return self.dropout1(x)








"""
Multi-Scale Attention Convolutional Neural Network (MSA-CNN) for EEG classification (sgoerttler).
"""

class MultiScaleConvolution(nn.Module):
    def __init__(self, config, num_filter_scales):
        super(MultiScaleConvolution, self).__init__()
        self.config = config

        convs1 = []
        dropouts1 = []
        if 'filter_scales_end' in config.keys():
            self.scale_indices = np.arange(config['filter_scales_start'] - 1, config['filter_scales_end'])
        elif 'num_filter_scales' in config.keys():
            self.scale_indices = np.arange(config['num_filter_scales'])

        # loop across scales
        for idx_scale, _ in enumerate(self.scale_indices):
            if self.config.get('multimodal_msm_conv1', False):
                conv1i = nn.Conv2d(self.config['num_channels'],
                                   self.config['out_channels_1'] * self.config['num_channels'],
                                   kernel_size=(1, self.config['kernel_1']),
                                   stride=(1, 1), padding='same', groups=self.config['num_channels'])
            else:
                num_out_channels = sum(((np.arange(self.config['out_channels_1'] * 4) // (
                        self.config['out_channels_1'] * 4 / num_filter_scales)) == idx_scale).astype(int))
                conv1i = nn.Conv2d(1, num_out_channels, kernel_size=(1, self.config['kernel_1']),
                                   stride=(1, 1), padding='same')

            convs1.append(conv1i)
            dropouts1.append(Dropout(config['dropout_rate']))

        self.convs1 = nn.ModuleList(convs1)
        self.dropouts1 = nn.ModuleList(dropouts1)

    def forward(self, x):
        x_scales = []

        if self.config.get('multimodal_msm_conv1', False):
            x = x.permute(0, 2, 1, 3)

        for idx_scale, scale in enumerate(2 ** self.scale_indices):
            xi = x.clone()

            xi = F.avg_pool2d(xi, (1, scale))
            xi = self.convs1[idx_scale](xi)
            xi = activation(xi, self.config)

            if self.config['complementary_pooling'] == 'max':
                xi = F.max_pool2d(xi, (1, 8 // scale))
            elif self.config['complementary_pooling'] == 'avg':
                xi = F.avg_pool2d(xi, (1, 8 // scale))

            xi = self.dropouts1[idx_scale](xi)

            if self.config.get('multimodal_msm_conv1', False):
                xi = xi.view(x.shape[0], self.config['num_channels'], self.config['out_channels_1'], -1)
                xi = xi.permute(0, 2, 1, 3).contiguous()

            x_scales.append(xi)

        return torch.cat(x_scales, dim=1)


class ScaleIntegrationConvolution(nn.Module):
    def __init__(self, config, num_filter_scales):
        super(ScaleIntegrationConvolution, self).__init__()
        self.config = config

        if self.config.get('multimodal_msm_conv2', False):
            self.conv_scales = nn.Conv2d(self.config['num_channels'],
                                         self.config['out_scales'] * self.config['num_channels'],
                                         kernel_size=(num_filter_scales * self.config['out_channels_1'],
                                                      self.config['kernel_scales']),
                                         stride=(1, self.config['kernel_scales']),
                                         groups=self.config['num_channels'])
        else:
            num_conv_scale_filters = 4 * self.config['out_channels_1']
            self.conv_scales = nn.Conv2d(num_conv_scale_filters,
                                         self.config['out_scales'],
                                         kernel_size=(1, self.config['kernel_scales']),
                                         stride=(1, self.config['kernel_scales']))
        self.dropout_scales = Dropout(self.config['dropout_rate'])

    def forward(self, x):
        if self.config.get('multimodal_msm_conv2', False):
            x = x.permute(0, 2, 1, 3)

        x = self.conv_scales(x)
        x = activation(x, self.config)
        x = self.dropout_scales(x)

        if self.config.get('multimodal_msm_conv2', False):
            x = x.view(x.shape[0], self.config['num_channels'], self.config['out_scales'], -1)
            x = x.permute(0, 2, 1, 3).contiguous()
        return x


class MultiScaleModule(nn.Module):
    def __init__(self, config):
        super(MultiScaleModule, self).__init__()
        self.config = config
        if 'num_filter_scales' in self.config.keys():
            self.num_filter_scales = self.config['num_filter_scales']
        elif 'filter_scales_end' in self.config.keys():
            self.num_filter_scales = self.config['filter_scales_end'] - self.config['filter_scales_start'] + 1

        self.multi_scale_convolution = MultiScaleConvolution(config, self.num_filter_scales)
        self.scale_integration_convolution = ScaleIntegrationConvolution(config, self.num_filter_scales)

    def forward(self, x):
        x = self.multi_scale_convolution(x)
        if self.config.get('return_conv1', False):
            return x  # optional: return for analysis
        return self.scale_integration_convolution(x)


class SpatialConvolution(nn.Module):
    def __init__(self, config):
        super(SpatialConvolution, self).__init__()
        self.config = config

        self.conv_spatial = nn.Conv2d(config['out_scales'], config['out_spatial'],
                                      kernel_size=(config['num_channels'], config['kernel_spatial']),
                                      stride=(1, 1), padding='valid')
        """
        global spatial convolution with no temporal dimension is equivalent to a linear layer with flattened input:
        self.conv_spatial = nn.Linear(config['num_channels'] * config['out_scales'], config['out_spatial'])
        in forward function:
        x = x.flatten(start_dim=1, end_dim=2).swapaxes(1, 2)
        x = self.conv_spatial(x)
        x = activation(x, self.config)
        return x.swapaxes(1, 2)
        """

    def forward(self, x):
        x = self.conv_spatial(x)
        x = activation(x, self.config)
        return x.squeeze(2)


class TemporalContextModule(nn.Module):
    def __init__(self, config, feature_dim, num_filter_scales):
        super(TemporalContextModule, self).__init__()
        num_heads = config['num_heads']
        num_layers = config['num_attention_layers']
        embed_dim = config['embedding_dim']
        dropout = config['dropout_rate']

        # manually compute sequence length after spatial convolution
        seq_length = config['length_time_series'] // (2 ** (num_filter_scales - 1)) - (
                config['kernel_scales'] - 1)

        # use linear layer to embed features, embedding is enabled if embedding dimension is above zero
        if embed_dim == 0:
            embed_dim = feature_dim
            self.embedding_flag = False
        else:
            self.embedding = nn.Linear(feature_dim, embed_dim)
            self.embedding_flag = True

        self.config = config
        if self.config['pos_encoding']:
            self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_length)

        # use custom transformer encoder layer only if access to attention weights required
        if config.get('access_attention_weights', False):
            encoder_layers = CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                           dim_feedforward=embed_dim * 2,
                                                           dropout=dropout, batch_first=True,
                                                           access_attention_weights=True,
                                                           config=config)
        else:
            encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                        dim_feedforward=embed_dim * 2, dropout=dropout,
                                                        batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers,
                                                         enable_nested_tensor=True)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, seq_length, feature_dim)

        if self.embedding_flag:
            x = self.embedding(x)  # (batch_size, seq_length, embed_dim)

        if self.config['pos_encoding']:
            x = self.pos_encoder(x)

        if self.config.get('access_attention_weights', False):
            return self.transformer_encoder(x)  # optional: return for analysis
        else:
            x = self.transformer_encoder(x)

        return x.transpose(1, 2)


class Mean(nn.Module):
    """Wrapper for torch.mean for clarity."""

    def __init__(self, dim):
        super(Mean, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)






class MSA_CNN(nn.Module):
    def __init__(self, config):
        super(MSA_CNN, self).__init__()
        self.config = config

        # scaling layer
        if config.get('input_scaling', False):
            self.scaling_layer = ScalingLayer(config['num_channels'])

        # multi-scale module
        self.msm = MultiScaleModule(config)

        # spatial layer
        self.spatial_layer = SpatialConvolution(config)
        out_dim = config['out_spatial']

        # temporal context module
        if config.get('num_attention_layers', 0) > 0:
            self.tcm = TemporalContextModule(config, out_dim, self.msm.num_filter_scales)
            out_dim = config['embedding_dim']

        # average across time
        self.time_average = Mean(dim=2)

        # fully connected layer and softmax
        self.fc = nn.Linear(out_dim, config['classes'])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.config.get('input_scaling', False):
            x = self.scaling_layer(x)

        x = self.msm(x)
        if self.config.get('return_conv1', False) or self.config.get('return_msm_conv2', False):
            return x  # optional: return for analysis

        x = self.spatial_layer(x)

        if self.config.get('num_attention_layers', 0) > 0:
            x = self.tcm(x)
            if self.config.get('access_attention_weights', False):
                return x  # optional: return for analysis

        x = self.time_average(x)

        x = self.fc(x)
        return self.softmax(x)





































import torch
import torch.nn as nn
import torch.nn.functional as F


class MSA_CNN_Sleep(nn.Module):
    """
    Simplified official MSA-CNN adapted for sleep staging.

    Input:
        x: (B, W, 1, 3000)

    Output:
        logits: (B, W, num_classes)
    """

    def __init__(
        self,
        num_classes=5,
        dropout=0.1,

        # official config
        num_filter_scales=4,
        kernel_1=15,
        out_channels_1=8,
        kernel_scales=5,
        out_scales=32,
        out_spatial=64,
        kernel_spatial=5,   # univariate official setting
        embedding_dim=32,
        num_heads=4,
        num_attention_layers=2,
    ):
        super().__init__()

        self.num_filter_scales = num_filter_scales
        self.out_channels_1 = out_channels_1
        self.out_scales = out_scales
        self.out_spatial = out_spatial
        self.embedding_dim = embedding_dim

        # ============================================================
        # Multi-scale convolution branch
        # ============================================================
        self.scale_pool = nn.ModuleList()
        self.scale_conv = nn.ModuleList()
        self.scale_postpool = nn.ModuleList()
        self.scale_dropout = nn.ModuleList()

        # official scales: [1,2,4,8]
        for i in range(num_filter_scales):
            scale = 2 ** i

            self.scale_pool.append(
                nn.AvgPool2d(kernel_size=(1, scale), stride=(1, scale))
            )

            self.scale_conv.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=out_channels_1,
                    kernel_size=(1, kernel_1),
                    padding="same",
                    bias=True,
                )
            )

            post_kernel = max(1, 8 // scale)

            self.scale_postpool.append(
                nn.MaxPool2d(kernel_size=(1, post_kernel))
            )

            self.scale_dropout.append(nn.Dropout(dropout))

        # ============================================================
        # Scale integration convolution
        # concat后通道数 = 4 * out_channels_1 = 32
        # ============================================================
        self.conv_scales = nn.Conv2d(
            in_channels=num_filter_scales * out_channels_1,
            out_channels=out_scales,
            kernel_size=(1, kernel_scales),
            stride=(1, kernel_scales),
            bias=True,
        )

        self.dropout_scales = nn.Dropout(dropout)

        # ============================================================
        # Spatial convolution
        # 对单通道输入，kernel height = 1
        # ============================================================
        self.conv_spatial = nn.Conv2d(
            in_channels=out_scales,
            out_channels=out_spatial,
            kernel_size=(1, kernel_spatial),
            bias=True,
        )

        # ============================================================
        # Transformer temporal context module
        # ============================================================
        self.embedding = nn.Linear(out_spatial, embedding_dim)

        self.pos_encoding = PositionalEncoding(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers
        )

        # classifier
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        x: (B, W, 1, 3000)
        """

        B, W, C, T = x.shape

        # flatten sequence dimension
        x = x.reshape(B * W, 1, 1, T)   # (BW,1,1,3000)

        # ============================================================
        # Multi-scale module
        # ============================================================
        multi_scale_feats = []

        for i in range(self.num_filter_scales):
            xi = self.scale_pool[i](x)            # avg pool by scale
            xi = self.scale_conv[i](xi)
            xi = F.relu(xi)
            xi = self.scale_postpool[i](xi)
            xi = self.scale_dropout[i](xi)
            multi_scale_feats.append(xi)

        x = torch.cat(multi_scale_feats, dim=1)

        # ============================================================
        # Scale integration conv
        # ============================================================
        x = self.conv_scales(x)
        x = F.relu(x)
        x = self.dropout_scales(x)

        # ============================================================
        # Spatial conv
        # ============================================================
        x = self.conv_spatial(x)
        x = F.relu(x)

        # remove spatial dim
        x = x.squeeze(2)   # (BW, out_spatial, L)

        # ============================================================
        # Temporal context transformer
        # ============================================================
        x = x.transpose(1, 2)            # (BW, L, out_spatial)
        x = self.embedding(x)            # (BW, L, embedding_dim)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        # temporal average
        x = x.mean(dim=1)                # (BW, embedding_dim)

        # classifier
        x = self.fc(x)                   # (BW, num_classes)

        # restore sequence dimension
        x = x.reshape(B, W, -1)          # (B, W, num_classes)

        return x










class TransformerEncoder_AttentionOnly(nn.Module):
    # 💡 修正 1：默认 depth 改为 1
    def __init__(self, embed_dim=128, depth=1, num_heads=8, dropout=0.1):
        super().__init__()

        # 💡 修正 2：必须加入 LayerNorm！
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              dropout=dropout,
                                              batch_first=True)
            })
            for _ in range(depth)
        ])

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """初始化所有注意力层和 LayerNorm 的参数"""
        for layer_dict in self.layers:
            attn_layer = layer_dict['attn']
            norm_layer = layer_dict['norm']

            # QKV 投影矩阵初始化
            nn.init.xavier_uniform_(attn_layer.in_proj_weight)
            if attn_layer.in_proj_bias is not None:
                nn.init.zeros_(attn_layer.in_proj_bias)

            # 输出投影矩阵初始化
            nn.init.xavier_uniform_(attn_layer.out_proj.weight)
            if attn_layer.out_proj.bias is not None:
                nn.init.zeros_(attn_layer.out_proj.bias)

            # LayerNorm 初始化
            nn.init.ones_(norm_layer.weight)
            nn.init.zeros_(norm_layer.bias)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        x: (B, N, D)
        """
        for layer_dict in self.layers:

            # 💡 修正 3：必须先过 LayerNorm，再进 Attention (Pre-Norm 结构最稳)
            x_norm = layer_dict['norm'](x)
            residual = x_norm
            attn_out, _ = layer_dict['attn'](x_norm, x_norm, x_norm,
                                             key_padding_mask=key_padding_mask,
                                             attn_mask=attn_mask)

            # 残差连接
            x = residual + self.dropout(attn_out)

        return x





# ============================================================
# MSA-CNN + Random Transformer
# ============================================================
class MSA_CNN_Sleep_RT(nn.Module):
    """
    MSA-CNN + Window-level Random Transformer

    Input:
        x: (B, W, 1, 3000)

    Output:
        logits: (B, W, num_classes)
    """

    def __init__(
        self,
        num_classes=5,
        dropout=0.1,

        # MSA-CNN config
        num_filter_scales=4,
        kernel_1=15,
        out_channels_1=8,
        kernel_scales=5,
        out_scales=32,
        out_spatial=64,
        kernel_spatial=5,

        # Transformer
        embedding_dim=32,
        num_heads=4,
        num_attention_layers=2,

        # RT
        rt_depth=1,
    ):
        super().__init__()

        self.num_filter_scales = num_filter_scales

        # ============================================================
        # Multi-scale convolution branch
        # ============================================================
        self.scale_pool = nn.ModuleList()
        self.scale_conv = nn.ModuleList()
        self.scale_postpool = nn.ModuleList()
        self.scale_dropout = nn.ModuleList()

        for i in range(num_filter_scales):

            scale = 2 ** i

            self.scale_pool.append(
                nn.AvgPool2d(
                    kernel_size=(1, scale),
                    stride=(1, scale)
                )
            )

            self.scale_conv.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=out_channels_1,
                    kernel_size=(1, kernel_1),
                    padding="same",
                    bias=True,
                )
            )

            post_kernel = max(1, 8 // scale)

            self.scale_postpool.append(
                nn.MaxPool2d(
                    kernel_size=(1, post_kernel)
                )
            )

            self.scale_dropout.append(
                nn.Dropout(dropout)
            )

        # ============================================================
        # Scale integration convolution
        # ============================================================
        self.conv_scales = nn.Conv2d(
            in_channels=num_filter_scales * out_channels_1,
            out_channels=out_scales,
            kernel_size=(1, kernel_scales),
            stride=(1, kernel_scales),
            bias=True,
        )

        self.dropout_scales = nn.Dropout(dropout)

        # ============================================================
        # Spatial convolution
        # ============================================================
        self.conv_spatial = nn.Conv2d(
            in_channels=out_scales,
            out_channels=out_spatial,
            kernel_size=(1, kernel_spatial),
            bias=True,
        )

        # ============================================================
        # Intra-epoch Transformer
        # ============================================================
        self.embedding = nn.Linear(
            out_spatial,
            embedding_dim
        )

        self.pos_encoding = PositionalEncoding(
            embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers
        )

        # ============================================================
        # Window-level Random Transformer
        # ============================================================
        self.random_transformer = TransformerEncoder_AttentionOnly(
            embed_dim=embedding_dim,
            depth=rt_depth,
            num_heads=num_heads,
            dropout=dropout,
        )

        # freeze RT
        for param in self.random_transformer.parameters():
            param.requires_grad = False

        # ============================================================
        # Classifier
        # ============================================================
        self.fc = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):
        """
                x: (B, W, 1, 3000)
                """

        B, W, C, T = x.shape

        # flatten sequence dimension
        x = x.reshape(B * W, 1, 1, T)  # (BW,1,1,3000)

        # ============================================================
        # Multi-scale module
        # ============================================================
        multi_scale_feats = []

        for i in range(self.num_filter_scales):
            xi = self.scale_pool[i](x)  # avg pool by scale
            xi = self.scale_conv[i](xi)
            xi = F.relu(xi)
            xi = self.scale_postpool[i](xi)
            xi = self.scale_dropout[i](xi)
            multi_scale_feats.append(xi)

        x = torch.cat(multi_scale_feats, dim=1)

        # ============================================================
        # Scale integration conv
        # ============================================================
        x = self.conv_scales(x)
        x = F.relu(x)
        x = self.dropout_scales(x)

        # ============================================================
        # Spatial conv
        # ============================================================
        x = self.conv_spatial(x)
        x = F.relu(x)

        # remove spatial dim
        x = x.squeeze(2)  # (BW, out_spatial, L)

        # ============================================================
        # Temporal context transformer
        # ============================================================
        x = x.transpose(1, 2)  # (BW, L, out_spatial)
        x = self.embedding(x)  # (BW, L, embedding_dim)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        # temporal average
        x = x.mean(dim=1)  # (BW, embedding_dim)

        # ============================================================
        # Restore sequence
        # ============================================================
        x = x.reshape(B, W, -1)

        # ============================================================
        # Inter-epoch Random Transformer
        # ============================================================
        x = self.random_transformer(x)

        # ============================================================
        # Classification
        # ============================================================
        x = self.fc(x)

        return x




