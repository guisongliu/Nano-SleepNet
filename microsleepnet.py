import torch
import math
import torch.nn as nn
from torch import Tensor
from torch.nn import functional
from scipy.optimize import minimize


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





class TransformerEncoder_AttentionOnly_random_test(nn.Module):
    def __init__(self, embed_dim=128, depth=1, num_heads=8, dropout=0.1):
        super().__init__()

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
            self._init_single_layer(layer_dict)

    def _init_single_layer(self, layer_dict):
        """单层初始化（供 forward 中重复调用）"""
        attn_layer = layer_dict['attn']
        norm_layer = layer_dict['norm']

        # QKV
        nn.init.xavier_uniform_(attn_layer.in_proj_weight)
        if attn_layer.in_proj_bias is not None:
            nn.init.zeros_(attn_layer.in_proj_bias)

        # 输出投影
        nn.init.xavier_uniform_(attn_layer.out_proj.weight)
        if attn_layer.out_proj.bias is not None:
            nn.init.zeros_(attn_layer.out_proj.bias)

        # LayerNorm
        nn.init.ones_(norm_layer.weight)
        nn.init.zeros_(norm_layer.bias)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        x: (B, N, D)
        """

        for layer_dict in self.layers:

            # ✅ 核心修改：eval 模式下每次 forward 都重新初始化
            if not self.training:
                self._init_single_layer(layer_dict)

            x_norm = layer_dict['norm'](x)
            residual = x_norm

            attn_out, _ = layer_dict['attn'](
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )

            x = residual + self.dropout(attn_out)

        return x



class TransformerEncoder_AttentionOnly_dk(nn.Module):
    # 💡 修正 1：默认 depth 改为 1
    def __init__(self, input_dim=128,embed_dim=128, depth=1, num_heads=8, dropout=0.1):
        super().__init__()
        # ------------------------------------------------------------
        # 显式输入投影层（关键新增）
        # ------------------------------------------------------------
        if input_dim != embed_dim:
            self.input_proj = nn.Linear(input_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()

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
        x = self.input_proj(x)
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

























#ECA+空间注意力机制模块
class ECA_SP_Layer(nn.Module):
    def __init__(self,c, b=1, gamma=2,spatial_kernel=7):#c代表channel
        super(ECA_SP_Layer, self).__init__()

        # #channel attention
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        #self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

        # spatial attention
        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        #7*7卷积核效果更好,conv前后feature map尺寸不变
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #ECA
        gap= self.avg_pool(x)#平均池化
        gap = self.conv1(gap.transpose(-1, -2)).transpose(-1, -2)#.transpose转置

        # gmp=self.max_pool(x)#最大池化
        # gmp = self.conv1(gmp.transpose(-1, -2)).transpose(-1, -2)

        channel_out = self.sigmoid(gap)#两池化结果直接相加
        x = channel_out * x#channel attention重标定feature map

        #CBAM spatial attention
        #按维度dim返回最大值以及最大值的索引
        max_out_sp, _ = torch.max(x, dim=1, keepdim=True)#dim=1表示取每一个特征点channel方向的最大值
        avg_out_sp = torch.mean(x, dim=1, keepdim=True)#keepdim=True表示输出和输入的维度一样
        attention_cat=torch.cat([max_out_sp, avg_out_sp], dim=1)#dim=1，沿着channel维度拼接
        spatial_out = self.sigmoid(self.conv(attention_cat))
        x = spatial_out * x#spatial attention与经channel attention重标定后的feature map相乘

        return x#返回经channel及spatial attention重标定以后的feature map


#定义基础eca+空间注意力卷积模块
class Base_eca_sp_conv(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride,padding,groups):
        super(Base_eca_sp_conv, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False#使用BN层，bias不起作用
            ),
            nn.BatchNorm1d(out_ch),#track_running_stats=False),#实际测试去掉BN层以后，收敛速度大幅下降
            nn.LeakyReLU(inplace=True),
            #nn.MaxPool1d(3)
            nn.MaxPool1d(3)

        )
        self.channel = ECA_SP_Layer(out_ch)

    def forward(self,x):
        y=self.conv(x)
        y=self.channel(y)
        return y




class Base_eca_sp_conv_residual(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, groups):
        super(Base_eca_sp_conv_residual, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(3)
        )

        self.channel = ECA_SP_Layer(out_ch)

        #  残差分支（自动对齐维度）
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
            nn.BatchNorm1d(out_ch),
            nn.MaxPool1d(3)
        )

    def forward(self, x):
        #identity = self.residual(x)   #  残差

        y = self.conv(x)
        y = self.channel(y)

        #y = y + identity             #  残差相加
        return y





#定义基础ECA卷积模块
class Base_conv_dilated(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride,padding,dilation):
        super(Base_conv_dilated, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False#使用BN层，bias不起作用
            ),
            nn.BatchNorm1d(out_ch),#实际测试去掉BN层以后，收敛速度大幅下降
            nn.LeakyReLU(inplace=True),
            #nn.MaxPool1d(3)
        )

    def forward(self,x):
        y=self.conv(x)
        return y


#定义 (无池化)基础eca+空间注意力卷积模块
class Base_eca_sp_conv_nonepool(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride,padding,groups):
        super(Base_eca_sp_conv_nonepool, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False#使用BN层，bias不起作用
            ),
            nn.BatchNorm1d(out_ch),#track_running_stats=False),#实际测试去掉BN层以后，收敛速度大幅下降
            nn.LeakyReLU(inplace=True),
            #nn.MaxPool1d(3)
            #nn.MaxPool1d(2)

        )
        self.channel = ECA_SP_Layer(out_ch)
    def forward(self,x):
        y=self.conv(x)
        y= self.channel(y)
        return y
#通道重排列
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels,  width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group,  width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1,  width)

    return x
#添加空洞卷积模块，进行特征提取以后的特征融合
class cnn_new(nn.Module):
    def __init__(self):
        super(cnn_new, self).__init__()#继承nn.Module中的代码
        self.conv1 = Base_eca_sp_conv(1,64,3,1,1,1)
        self.conv2 = Base_eca_sp_conv(64,128,3,1,1,64)
        self.conv3 = Base_eca_sp_conv(128,128,3,1,1,64)
        self.conv4 = Base_eca_sp_conv(128,128,3,1,1,64)
        self.conv5 = Base_eca_sp_conv_nonepool(128,128,3,1,1,64)

        self.dropout = nn.Dropout(p=0.5)

        self.dila_conv1 = Base_conv_dilated(128,32,3,1,1,1)
        self.dila_conv2 = Base_conv_dilated(32,64,3,1,2,2)
        self.dila_conv3 = Base_conv_dilated(64,128,3,1,4,4)

        self.out = nn.Linear(128, 5)
        #self.feas = []



    def forward(self, x):
        x = self.conv1(x)#input [bz,channel,feature][bz,1,3000]
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)#output[bz,128,37]

        x=self.dropout(x)

        x = self.dila_conv1(x)
        x = self.dila_conv2(x)
        x = self.dila_conv3(x)

        #self.feas.append(x)

        x = functional.adaptive_avg_pool1d(x, 1)#全局平均池化层
        x = x.view(x.shape[0], -1)#通过view函数将张量x变成batchsize行，4* 4* 512列，-1表示自动计算列数

        x = self.out(x)
        return x













class microsleepnet_RT(nn.Module):
    def __init__(self, window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet_RT, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        # self.dila_conv1 = Base_conv_dilated(128, 32, 3, 1, 1, 1)
        # self.dila_conv2 = Base_conv_dilated(32, 64, 3, 1, 2, 2)
        # self.dila_conv3 = Base_conv_dilated(64, 128, 3, 1, 4, 4)

        # ==========================================
        # 2. 新增：时序建模模块 (Random Transformer)
        # ==========================================
        # 可训练的时间位置编码
        #self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        # # 随机 Transformer Encoder
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=0.1,
        #     activation='gelu',
        #     batch_first=True
        # )
        #self.random_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.random_transformer = TransformerEncoder_AttentionOnly(embed_dim=128, depth=num_layers, num_heads=nhead,
                                                         dropout=0.1)

        # 初始化并冻结 RT，严格遵循 RAPK 理论
        #self._init_random_transformer()

        # for param in self.random_transformer.parameters():
        #     param.requires_grad = False

        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)

    # def _init_random_transformer(self):
    #     """严格按照 RAPK 理论使用 Xavier Uniform 初始化"""
    #     for layer in self.random_transformer.layers:
    #         nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
    #         if layer.self_attn.in_proj_bias is not None:
    #             nn.init.zeros_(layer.self_attn.in_proj_bias)
    #         nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
    #         if layer.self_attn.out_proj.bias is not None:
    #             nn.init.zeros_(layer.self_attn.out_proj.bias)
    #
    #         nn.init.xavier_uniform_(layer.linear1.weight)
    #         nn.init.zeros_(layer.linear1.bias)
    #         nn.init.xavier_uniform_(layer.linear2.weight)
    #         nn.init.zeros_(layer.linear2.bias)
    #
    #         nn.init.ones_(layer.norm1.weight)
    #         nn.init.zeros_(layer.norm1.bias)
    #         nn.init.ones_(layer.norm2.weight)
    #         nn.init.zeros_(layer.norm2.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)
        #
        # x = self.dila_conv1(x)
        # x = self.dila_conv2(x)
        # x = self.dila_conv3(x)  # 此时形状: [B*W, 128, L]

        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 4. 时序建模 (Random Transformer)
        #x = x + self.pos_embed  # 加上位置编码
        x = self.random_transformer(x)  # 时序平滑，输出: [B, W, 128]

        #x = self.dropout(x)

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x












class microsleepnet_dilated_RT(nn.Module):
    def __init__(self, window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet_dilated_RT, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        self.dila_conv1 = Base_conv_dilated(128, 32, 3, 1, 1, 1)
        self.dila_conv2 = Base_conv_dilated(32, 64, 3, 1, 2, 2)
        self.dila_conv3 = Base_conv_dilated(64, 128, 3, 1, 4, 4)


        self.random_transformer = TransformerEncoder_AttentionOnly(embed_dim=128, depth=num_layers, num_heads=nhead,
                                                         dropout=0.1)

        # 初始化并冻结 RT，严格遵循 RAPK 理论
        #self._init_random_transformer()

        # for param in self.random_transformer.parameters():
        #     param.requires_grad = False

        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)

        x = self.dila_conv1(x)
        x = self.dila_conv2(x)
        x = self.dila_conv3(x)  # 此时形状: [B*W, 128, L]

        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 4. 时序建模 (Random Transformer)
        #x = x + self.pos_embed  # 加上位置编码
        x = self.random_transformer(x)  # 时序平滑，输出: [B, W, 128]

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x


















class microsleepnet_dilated_RT_dk(nn.Module):
    def __init__(self, window_size=10, d_model=512, nhead=4, num_layers=1):
        super(microsleepnet_dilated_RT_dk, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        self.dila_conv1 = Base_conv_dilated(128, 32, 3, 1, 1, 1)
        self.dila_conv2 = Base_conv_dilated(32, 64, 3, 1, 2, 2)
        self.dila_conv3 = Base_conv_dilated(64, 128, 3, 1, 4, 4)

        # ==========================================
        # 2. 新增：时序建模模块 (Random Transformer)
        # ==========================================


        self.random_transformer = TransformerEncoder_AttentionOnly_dk(input_dim=128,embed_dim=d_model, depth=num_layers, num_heads=nhead,
                                                         dropout=0.1)

        # 初始化并冻结 RT，严格遵循 RAPK 理论
        for param in self.random_transformer.layers.parameters():
            param.requires_grad = False

        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)

        x = self.dila_conv1(x)
        x = self.dila_conv2(x)
        x = self.dila_conv3(x)  # 此时形状: [B*W, 128, L]

        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 4. 时序建模 (Random Transformer)
        #x = x + self.pos_embed  # 加上位置编码
        x = self.random_transformer(x)  # 时序平滑，输出: [B, W, 128]

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x








class microsleepnet_encoder_RT_dk(nn.Module):
    def __init__(self, window_size=10, d_model=512, nhead=4, num_layers=1):
        super(microsleepnet_encoder_RT_dk, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)



        # ==========================================
        # 2. 新增：时序建模模块 (Random Transformer)
        # ==========================================

        self.random_transformer = TransformerEncoder_AttentionOnly_dk(input_dim=128, embed_dim=d_model,
                                                                      depth=num_layers, num_heads=nhead,
                                                                      dropout=0.1)

        # 初始化并冻结 RT，严格遵循 RAPK 理论
        for param in self.random_transformer.layers.parameters():
            param.requires_grad = False

        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)



        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 4. 时序建模 (Random Transformer)

        x = self.random_transformer(x)  # 时序平滑，输出: [B, W, 128]

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x


class microsleepnet_RT_dk(nn.Module):
    def __init__(self, window_size=10, d_model=512, nhead=4, num_layers=1):
        super(microsleepnet_RT_dk, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)


        # ==========================================
        # 2. 新增：时序建模模块 (Random Transformer)
        # ==========================================

        self.random_transformer = TransformerEncoder_AttentionOnly_dk(input_dim=128,embed_dim=d_model, depth=num_layers, num_heads=nhead,
                                                         dropout=0.1)

        # 初始化并冻结 RT，严格遵循 RAPK 理论
        for param in self.random_transformer.layers.parameters():
            param.requires_grad = False

        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)



        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 4. 时序建模 (Random Transformer)
        x = self.random_transformer(x)  # 时序平滑，输出: [B, W, 128]

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x






class microsleepnet_dilated_trainT(nn.Module):
    def __init__(self, window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet_dilated_trainT, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        self.dila_conv1 = Base_conv_dilated(128, 32, 3, 1, 1, 1)
        self.dila_conv2 = Base_conv_dilated(32, 64, 3, 1, 2, 2)
        self.dila_conv3 = Base_conv_dilated(64, 128, 3, 1, 4, 4)

        # ==========================================
        # 2. 新增：时序建模模块 (Random Transformer)
        # ==========================================
        # 可训练的时间位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        # 随机 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.random_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #self.random_transformer = TransformerEncoder_AttentionOnly(embed_dim=128, depth=num_layers, num_heads=nhead,
                                                         # dropout=0.1)

        # 初始化并冻结 RT，严格遵循 RAPK 理论
        #self._init_random_transformer()
        # for param in self.random_transformer.parameters():
        #     param.requires_grad = False

        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)

    def _init_random_transformer(self):
        """严格按照 RAPK 理论使用 Xavier Uniform 初始化"""
        for layer in self.random_transformer.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            if layer.self_attn.out_proj.bias is not None:
                nn.init.zeros_(layer.self_attn.out_proj.bias)

            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)

            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)

        x = self.dila_conv1(x)
        x = self.dila_conv2(x)
        x = self.dila_conv3(x)  # 此时形状: [B*W, 128, L]

        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 4. 时序建模 (Random Transformer)
        x = x + self.pos_embed  # 加上位置编码
        x = self.random_transformer(x)  # 时序平滑，输出: [B, W, 128]

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x












class microsleepnet_trainT(nn.Module):
    def __init__(self, window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet_trainT, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        # self.dila_conv1 = Base_conv_dilated(128, 32, 3, 1, 1, 1)
        # self.dila_conv2 = Base_conv_dilated(32, 64, 3, 1, 2, 2)
        # self.dila_conv3 = Base_conv_dilated(64, 128, 3, 1, 4, 4)

        # ==========================================
        # 2. 新增：时序建模模块 (Random Transformer)
        # ==========================================
        # 可训练的时间位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        # 随机 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.random_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        # 初始化并冻结 RT，严格遵循 RAPK 理论
        self._init_random_transformer()
        # for param in self.random_transformer.parameters():
        #     param.requires_grad = False

        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)

    def _init_random_transformer(self):
        """严格按照 RAPK 理论使用 Xavier Uniform 初始化"""
        for layer in self.random_transformer.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            if layer.self_attn.out_proj.bias is not None:
                nn.init.zeros_(layer.self_attn.out_proj.bias)

            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)

            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)


        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 4. 时序建模 (Random Transformer)
        x = x + self.pos_embed  # 加上位置编码
        x = self.random_transformer(x)  # 时序平滑，输出: [B, W, 128]

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x










class microsleepnet_dilated_LSTM(nn.Module):
    def __init__(self, window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet_dilated_LSTM, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        self.dila_conv1 = Base_conv_dilated(128, 32, 3, 1, 1, 1)
        self.dila_conv2 = Base_conv_dilated(32, 64, 3, 1, 2, 2)
        self.dila_conv3 = Base_conv_dilated(64, 128, 3, 1, 4, 4)

        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, dropout=0.5)
        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)

        x = self.dila_conv1(x)
        x = self.dila_conv2(x)
        x = self.dila_conv3(x)  # 此时形状: [B*W, 128, L]

        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        #4. LSTM时序建模
        x, _ = self.rnn(x)

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x







class microsleepnet_encoder_LSTM(nn.Module):
    def __init__(self, window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet_encoder_LSTM, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, dropout=0.5)
        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)


        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        #4. LSTM时序建模
        x, _ = self.rnn(x)

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x


class microsleepnet_dilated_GRU(nn.Module):
    def __init__(self, window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet_dilated_GRU, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        self.dila_conv1 = Base_conv_dilated(128, 32, 3, 1, 1, 1)
        self.dila_conv2 = Base_conv_dilated(32, 64, 3, 1, 2, 2)
        self.dila_conv3 = Base_conv_dilated(64, 128, 3, 1, 4, 4)

        self.rnn = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, dropout=0.5)
        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)

        x = self.dila_conv1(x)
        x = self.dila_conv2(x)
        x = self.dila_conv3(x)  # 此时形状: [B*W, 128, L]

        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 4. LSTM时序建模
        x, _ = self.rnn(x)

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x







class microsleepnet_encoder_GRU(nn.Module):
    def __init__(self, window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet_encoder_GRU, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        self.rnn = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, dropout=0.5)
        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)

        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 4. LSTM时序建模
        x, _ = self.rnn(x)

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x






class microsleepnet(nn.Module):
    def __init__(self, num_classes,window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        self.dila_conv1 = Base_conv_dilated(128, 32, 3, 1, 1, 1)
        self.dila_conv2 = Base_conv_dilated(32, 64, 3, 1, 2, 2)
        self.dila_conv3 = Base_conv_dilated(64, 128, 3, 1, 4, 4)

        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)

        x = self.dila_conv1(x)
        x = self.dila_conv2(x)
        x = self.dila_conv3(x)  # 此时形状: [B*W, 128, L]

        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x



class microsleepnet_encoder(nn.Module):
    def __init__(self, window_size=10, d_model=128, nhead=4, num_layers=1):
        super(microsleepnet_encoder, self).__init__()

        # ==========================================
        # 1. 原始的 CNN 结构 (充当 Epoch Encoder)
        # ==========================================
        self.conv1 = Base_eca_sp_conv(1, 64, 3, 1, 1, 1)
        self.conv2 = Base_eca_sp_conv(64, 128, 3, 1, 1, 64)
        self.conv3 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv4 = Base_eca_sp_conv(128, 128, 3, 1, 1, 64)
        self.conv5 = Base_eca_sp_conv_nonepool(128, 128, 3, 1, 1, 64)

        self.dropout = nn.Dropout(p=0.5)

        # self.dila_conv1 = Base_conv_dilated(128, 32, 3, 1, 1, 1)
        # self.dila_conv2 = Base_conv_dilated(32, 64, 3, 1, 2, 2)
        # self.dila_conv3 = Base_conv_dilated(64, 128, 3, 1, 4, 4)

        # ==========================================
        # 2. 新增：时序建模模块 (Random Transformer)
        # ==========================================
        # 可训练的时间位置编码
        #self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        # # 随机 Transformer Encoder
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=0.1,
        #     activation='gelu',
        #     batch_first=True
        # )
        #self.random_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #self.random_transformer = TransformerEncoder_AttentionOnly(embed_dim=128, depth=num_layers, num_heads=nhead,
                                                         # dropout=0.1)

        # 初始化并冻结 RT，严格遵循 RAPK 理论
        #self._init_random_transformer()
        # for param in self.random_transformer.parameters():
        #     param.requires_grad = False

        # ==========================================
        # 3. 分类头
        # ==========================================
        self.out = nn.Linear(d_model, 5)

    # def _init_random_transformer(self):
    #     """严格按照 RAPK 理论使用 Xavier Uniform 初始化"""
    #     for layer in self.random_transformer.layers:
    #         nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
    #         if layer.self_attn.in_proj_bias is not None:
    #             nn.init.zeros_(layer.self_attn.in_proj_bias)
    #         nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
    #         if layer.self_attn.out_proj.bias is not None:
    #             nn.init.zeros_(layer.self_attn.out_proj.bias)
    #
    #         nn.init.xavier_uniform_(layer.linear1.weight)
    #         nn.init.zeros_(layer.linear1.bias)
    #         nn.init.xavier_uniform_(layer.linear2.weight)
    #         nn.init.zeros_(layer.linear2.bias)
    #
    #         nn.init.ones_(layer.norm1.weight)
    #         nn.init.zeros_(layer.norm1.bias)
    #         nn.init.ones_(layer.norm2.weight)
    #         nn.init.zeros_(layer.norm2.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入形状必须为 [Batch, Window_size, Channel, Time]
                              例如: [B, 10, 1, 3000]
        """
        # 1. 维度折叠：将 Batch 和 Window 合并，批量送入 CNN
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)  # 变成 [B*10, 1, 3000]

        # 2. 原始 CNN 前向传播 (Epoch Encoder)
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, 32)
        x = self.conv3(x)
        x = channel_shuffle(x, 64)
        x = self.conv4(x)
        x = channel_shuffle(x, 64)
        x = self.conv5(x)

        x = self.dropout(x)
        #
        # x = self.dila_conv1(x)
        # x = self.dila_conv2(x)
        # x = self.dila_conv3(x)  # 此时形状: [B*W, 128, L]

        # 3. Epoch 内特征聚合 (保留这一步，获取单 epoch 向量)
        x = functional.adaptive_avg_pool1d(x, 1)  # 全局平均池化层
        x = x.view(B, W, -1)  # 维度展开恢复序列形状: [B, W, 128]

        # # 4. 时序建模 (Random Transformer)
        # x = x + self.pos_embed  # 加上位置编码
        # x = self.random_transformer(x)  # 时序平滑，输出: [B, W, 128]

        # 5. 分类输出
        x = self.out(x)  # 形状: [B, W, 5]

        return x








import numpy as np
from scipy.stats import mode
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, f1_score


class SequenceSmootherNonOverlapWindow:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes

    # -------------------------------------------------
    # 1 Moving Average (Non-overlap)
    # -------------------------------------------------
    def moving_average(self, probs, window_size=10):
        n_samples = probs.shape[0]
        smoothed_probs = np.zeros_like(probs)
        for start in range(0, n_samples, window_size):
            end = min(start + window_size, n_samples)
            win = probs[start:end].copy()
            mean_prob = np.mean(win, axis=0)
            smoothed_probs[start:end] = mean_prob
        return np.argmax(smoothed_probs, axis=1)

    # -------------------------------------------------
    # 2 Majority Voting / Median Filter (Non-overlap)
    # -------------------------------------------------
    def majority_voting(self, preds, window_size=10):
        n_samples = len(preds)
        smoothed_preds = np.zeros_like(preds)
        for start in range(0, n_samples, window_size):
            end = min(start + window_size, n_samples)
            win = preds[start:end].copy()
            m = mode(win, keepdims=False)[0]
            smoothed_preds[start:end] = m
        return smoothed_preds

    # -------------------------------------------------
    # 3 Exponential Smoothing (windowed non-overlap)
    # -------------------------------------------------
    def exponential_smoothing(self, probs, window_size=10, alpha=0.3):
        n_samples = probs.shape[0]
        smoothed_probs = np.zeros_like(probs)
        for start in range(0, n_samples, window_size):
            end = min(start + window_size, n_samples)
            win = probs[start:end].copy()
            # init first step
            y = win[0].copy()
            for t in range(1, len(win)):
                y = alpha * win[t] + (1 - alpha) * y
                win[t] = y
            smoothed_probs[start:end] = win
        return np.argmax(smoothed_probs, axis=1)

    # -------------------------------------------------
    # 4 Gaussian Smoothing (windowed non-overlap)
    # -------------------------------------------------
    def gaussian_smoothing(self, probs, window_size=10, sigma_ratio=0.3):
        n_samples = probs.shape[0]
        smoothed_probs = np.zeros_like(probs)
        for start in range(0, n_samples, window_size):
            end = min(start + window_size, n_samples)
            win = probs[start:end].copy()
            sigma = max(1, int(len(win) * sigma_ratio))
            smoothed_win = gaussian_filter1d(win, sigma=sigma, axis=0)
            smoothed_probs[start:end] = smoothed_win
        return np.argmax(smoothed_probs, axis=1)

    # -------------------------------------------------
    # 5 Savitzky-Golay Smoothing (windowed non-overlap)
    # -------------------------------------------------
    def savgol_smoothing(self, probs, window_size=10, polyorder=2):
        n_samples = probs.shape[0]
        smoothed_probs = np.zeros_like(probs)
        for start in range(0, n_samples, window_size):
            end = min(start + window_size, n_samples)
            win = probs[start:end].copy()
            # window_length must be odd and <= len(win)
            wlen = min(window_size, len(win))
            if wlen % 2 == 0:
                wlen -= 1
            if wlen < 3:  # too small window
                smoothed_probs[start:end] = win
                continue
            smoothed_win = savgol_filter(win, window_length=wlen, polyorder=polyorder, axis=0)
            smoothed_probs[start:end] = smoothed_win
        return np.argmax(smoothed_probs, axis=1)

    # -------------------------------------------------
    # 6 Weighted Moving Average (windowed non-overlap)
    # -------------------------------------------------
    def weighted_moving_average(self, probs, weights):
        """
        probs: shape (n_samples, n_classes)
        weights: 1D array, 标准权重模板，长度 k
        """
        n_samples, n_classes = probs.shape
        k = len(weights)
        half = k // 2
        smoothed_probs = np.zeros_like(probs)

        for i in range(n_samples):
            # 实际窗口范围
            start = max(0, i - half)
            end = min(n_samples, i + half + 1)
            window = probs[start:end]

            # 根据窗口长度生成权重
            win_len = end - start
            if win_len != k:
                # 对边缘窗口重新生成权重
                w_adjusted = np.hanning(win_len)
                w_adjusted = w_adjusted / (w_adjusted.sum() + 1e-8)
            else:
                w_adjusted = weights

            # 广播相乘并求和
            smoothed_probs[i] = np.sum(window * w_adjusted[:, None], axis=0)

        return np.argmax(smoothed_probs, axis=1)


    # -------------------------------------------------
    # 7 Kalman Filter Smoothing
    # -------------------------------------------------
    # def kalman_smoothing(self, probs, process_var=1e-4, obs_var=1e-2):
    #     """
    #     对每个类别概率序列独立进行一维 Kalman 滤波。
    #
    #     Args:
    #         probs: ndarray, shape (n_samples, n_classes)
    #         process_var: 状态转移噪声 Q
    #         obs_var: 观测噪声 R
    #
    #     Returns:
    #         pred: ndarray, shape (n_samples,)
    #     """
    #     n_samples, n_classes = probs.shape
    #     smoothed = np.zeros_like(probs)
    #
    #     for c in range(n_classes):
    #         z = probs[:, c]
    #
    #         # 初始化
    #         x_est = z[0]
    #         P = 1.0
    #
    #         smoothed[0, c] = x_est
    #
    #         for t in range(1, n_samples):
    #             # Prediction
    #             x_pred = x_est
    #             P_pred = P + process_var
    #
    #             # Update
    #             K = P_pred / (P_pred + obs_var)
    #             x_est = x_pred + K * (z[t] - x_pred)
    #             P = (1 - K) * P_pred
    #
    #             smoothed[t, c] = x_est
    #
    #     # 防止数值问题，重新归一化
    #     smoothed = np.clip(smoothed, 1e-8, None)
    #     smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)
    #
    #     return np.argmax(smoothed, axis=1)

    def kalman_smoothing(self, probs, window_size=5, process_var=1e-4, obs_var=1e-2):
        n_samples, n_classes = probs.shape
        smoothed = np.zeros_like(probs)
        w = window_size // 2

        for c in range(n_classes):
            z = probs[:, c]
            for t in range(n_samples):
                # 取窗口内的片段
                start = max(0, t - w)
                end = min(n_samples, t + w + 1)
                segment = z[start:end]

                # 在窗口内做卡尔曼
                x_est = segment[0]
                P = 1.0
                for s in range(1, len(segment)):
                    x_pred = x_est
                    P_pred = P + process_var
                    K = P_pred / (P_pred + obs_var)
                    x_est = x_pred + K * (segment[s] - x_pred)
                    P = (1 - K) * P_pred

                smoothed[t, c] = x_est

        smoothed = np.clip(smoothed, 1e-8, None)
        smoothed /= smoothed.sum(axis=1, keepdims=True)
        return np.argmax(smoothed, axis=1)

    # -------------------------------------------------
    # 8 Total Variation Denoising
    # -------------------------------------------------
    def tv_denoising(self, probs, lam=0.1):
        """
        对每个类别概率序列进行 Total Variation Denoising.

        min_x 0.5 * ||x - y||^2 + lam * TV(x)

        Args:
            probs: ndarray, shape (n_samples, n_classes)
            lam: TV 正则强度，越大越平滑

        Returns:
            pred: ndarray, shape (n_samples,)
        """
        n_samples, n_classes = probs.shape
        smoothed = np.zeros_like(probs)

        for c in range(n_classes):
            y = probs[:, c].copy()

            def objective(x):
                fidelity = 0.5 * np.sum((x - y) ** 2)
                tv = lam * np.sum(np.abs(np.diff(x)))
                return fidelity + tv

            # 初始值使用原序列
            x0 = y.copy()

            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * n_samples,
            )

            smoothed[:, c] = res.x

        # 重新归一化
        smoothed = np.clip(smoothed, 1e-8, None)
        smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)

        return np.argmax(smoothed, axis=1)

    # # -------------------------------------------------
    # # evaluation
    # # -------------------------------------------------
    # def evaluate(self, y_true, y_pred, name="Method"):
    #     acc = accuracy_score(y_true, y_pred)
    #     f1 = f1_score(y_true, y_pred, average="weighted")
    #     print(f"[{name:<25}] ACC: {acc*100:.4f}% | F1: {f1*100:.4f}%")
    #     return acc, f1











#固定注意力
class GlobalBlindSmoother(nn.Module):
    def __init__(self):
        super().__init__()
        # # # 1. 时间位置编码（可学习参数）
        # self.time_pos = nn.Parameter(torch.zeros((1, window_size, 128)))  # [1, seq_len, embed_dim]
        #
        # nn.init.trunc_normal_(self.time_pos, std=0.02)  # 初始化为截断正态分布

    def forward(self, x_in_t_x: torch.Tensor, pretrain: bool = False):
        """
        Args:
            x_in_t_x: 输入张量 [Batch, Seq_Len, Feature_Dim]
                      假设 Seq_Len = 10 (即 T=10)
        Returns:
            平滑后的张量，形状不变。
            但注意：输出的每个时间步的值都会变得完全一样。
        """
        # pos = self.time_pos[:, :x_in_t_x.size(1), :]
        # x_in_t_x = x_in_t_x + pos  # [N, 10, 128]

        B, T, D = x_in_t_x.shape
        device = x_in_t_x.device

        # 1. 构建全均匀注意力矩阵 (Global Uniform Matrix)
        # 形状: [T, T]
        # 每一个元素的值都是 1/T
        # 比如 T=10，那么矩阵里所有元素都是 0.1
        attn_weights = torch.ones(T, T, device=device) / T

        # 2. 执行矩阵乘法 (Matrix Multiplication)
        # [T, T] @ [B, T, D] -> [B, T, D] (PyTorch会自动广播Batch维度)
        #
        # 计算逻辑：
        # Output[t] = Sum(attn_weights[t, k] * Input[k]) for k in 0..T-1
        #           = Sum( (1/T) * Input[k] )
        #           = Mean(Input)
        x_out = torch.matmul(attn_weights, x_in_t_x)

        return x_out
