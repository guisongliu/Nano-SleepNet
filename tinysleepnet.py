import torch
import torch.nn as nn
import torch.nn.functional as F
from zuwangdaima import TransformerEncoder_AttentionOnly

# ===============================
# 🔹 CNN Backbone（完全对齐原版）
# ===============================
class TinySleepNetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        fs = 100
        first_filter_size = int(fs / 2.0)
        first_stride = int(fs / 16.0)

        self.conv1 = nn.Conv1d(1, 128, kernel_size=first_filter_size, stride=first_stride)
        self.bn1 = nn.BatchNorm1d(128)

        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)

        self.conv2_1 = nn.Conv1d(128, 128, kernel_size=8, stride=1)
        self.bn2_1 = nn.BatchNorm1d(128)

        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=8, stride=1)
        self.bn2_2 = nn.BatchNorm1d(128)

        self.conv2_3 = nn.Conv1d(128, 128, kernel_size=8, stride=1)
        self.bn2_3 = nn.BatchNorm1d(128)

        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        x: (B, W, 1, 3000)
        return: (B, W, F)
        """

        B, W, C, T = x.shape

        # ------------------------------------------------
        # 1. flatten window维度
        # ------------------------------------------------
        x = x.reshape(B * W, C, T)   # (B*W, 1, 3000)

        # ------------------------------------------------
        # 2. CNN feature extractor
        # ------------------------------------------------
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn2_3(self.conv2_3(x)))

        x = self.pool2(x)

        # (B*W, F, L)
        x = x.mean(dim=-1)  # ⭐关键：变成 (B*W, F)

        #x = self.dropout(x)

        # ------------------------------------------------
        # 3. restore sequence
        # ------------------------------------------------
        x = x.reshape(B, W, -1)  # (B, W, F)

        return x


# ===============================
# 🔹 TinySleepNet（完整版本）
# ===============================
class TinySleepNet(nn.Module):
    def __init__(self, num_classes,use_rnn=False):
        super().__init__()

        self.use_rnn = use_rnn

        # CNN
        self.cnn = TinySleepNetCNN()
        #self.rnn =nn.LSTM(input_size=128, hidden_size=128, num_layers=1,batch_first=True,dropout=0.5)
        self.dropout = nn.Dropout(0.5)

        # FC（softmax_linear）
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, seq_lengths=None):
        """
        x: (B, W, 1, 1)
        return: (B, W, num_classes)
        """

        # -------------------------
        # 1. CNN feature extraction
        # -------------------------
        x = self.cnn(x)  # (B, W, F)




        #x = self.dropout(x)
        # -------------------------
        # 2. RNN sequence modeling
        # -------------------------

        # if self.use_rnn:
        #     x, _ = self.rnn(x)

        #x = x.flatten(1)
        # -------------------------
        # 3. classifier（关键：保持时间维）
        # -------------------------

        #x = self.dropout(x)

        x = self.fc(x)  # (B, W, C)

        return x


# ===============================
# 🔹 Loss（完全对齐原版）
# ===============================
class TinySleepNetLoss(nn.Module):
    def __init__(self, config, use_rnn=False):
        super().__init__()
        self.use_rnn = use_rnn
        self.class_weights = torch.tensor(config["class_weights"], dtype=torch.float32)

    def forward(self, logits, labels, loss_weights=None):
        ce = F.cross_entropy(logits, labels, reduction='none')

        if self.use_rnn:
            # 序列权重
            ce = ce * loss_weights

            # 类别权重
            class_w = self.class_weights.to(logits.device)
            sample_w = class_w[labels]
            ce = ce * sample_w

            loss = ce.sum() / loss_weights.sum()
        else:
            loss = ce.mean()

        return loss







class TinySleepNet_Encoder_RT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # CNN
        self.cnn = TinySleepNetCNN(config)

        #self.dropout = nn.Dropout(0.5)

        self.random_transformer = TransformerEncoder_AttentionOnly(embed_dim=128, depth=1, num_heads=4,
                                                                   dropout=0.1)

        # 初始化并冻结 RT，严格遵循 RAPK 理论
        for param in self.random_transformer.parameters():
            param.requires_grad = False

        # FC（softmax_linear）
        self.fc = nn.Linear(128, config["n_classes"])

    def forward(self, x):
        """
        x: (B, W, 1, 1)
        return: (B, W, num_classes)
        """

        # -------------------------
        # 1. CNN feature extraction
        # -------------------------
        x = self.cnn(x)  # (B, W, F)

        x = self.random_transformer(x)  # 时序平滑，输出: [B, W, 128]

        #x = x.flatten(1)
        # -------------------------
        # 3. classifier（关键：保持时间维）
        # -------------------------
        #x = self.dropout(x)
        x = self.fc(x)  # (B, W, C)

        return x











class TinySleepNet_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # CNN
        self.cnn = TinySleepNetCNN(config)

        #self.dropout = nn.Dropout(0.5)

        # FC（softmax_linear）
        self.fc = nn.Linear(128, config["n_classes"])

    def forward(self, x):
        """
        x: (B, W, 1, 1)
        return: (B, W, num_classes)
        """

        # -------------------------
        # 1. CNN feature extraction
        # -------------------------
        x = self.cnn(x)  # (B, W, F)

        #x = x.flatten(1)
        # -------------------------
        # 3. classifier（关键：保持时间维）
        # -------------------------
        #x = self.dropout(x)
        x = self.fc(x)  # (B, W, C)

        return x
