# -*- coding: utf-8 -*-
# Modify from DeepSleepNet
import torch
import torch.nn as nn
from zuwangdaima import TransformerEncoder_AttentionOnly


class deepsleepnet_Encoder(nn.Module):
    def __init__(self, dropout):
        super(deepsleepnet_Encoder, self).__init__()
        self.encoder_branch1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=49, stride=6, padding=24),  # 3000->500
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),  # 500->63

            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding='same'),  # 63->63
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=9, stride=1, padding='same'),  # 63->63
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding='same'),  # 63->63
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),  # 63->16
            nn.AdaptiveAvgPool1d(1),  # 16->1
        )

        self.encoder_branch2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50),  # 3000->53
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4),  # 53->13

            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding='same'),  # 13->13
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding='same'),  # 13->13
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding='same'),  # 13->13
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2),  # 13->6
            nn.AdaptiveAvgPool1d(1),  # 6->1
        )

    def forward(self, x):
        # x: (B, W, 1, 3000)
        B, W, C, L = x.shape

        # 👉 展平时间维
        x = x.reshape(B * W, C, L)  # (B*W, 1, 3000)

        # 👉 原 encoder
        x1 = self.encoder_branch1(x).squeeze(-1)  # (B*W, 256)
        x2 = self.encoder_branch2(x).squeeze(-1)  # (B*W, 256)

        x = torch.cat([x1, x2], dim=-1)  # (B*W, 512)

        # 👉 还原回序列结构
        x = x.reshape(B, W, -1)  # (B, W, 512)

        return x






class deepSleepNet_Encoder_RT(nn.Module):
    def __init__(self,):
        super().__init__()


        # CNN
        self.cnn = deepsleepnet_Encoder(0.5)

        #self.dropout = nn.Dropout(0.5)

        self.random_transformer = TransformerEncoder_AttentionOnly(embed_dim=512, depth=1, num_heads=4,
                                                                   dropout=0.1)

        # 初始化并冻结 RT，严格遵循 RAPK 理论
        for param in self.random_transformer.parameters():
            param.requires_grad = False

        # FC（softmax_linear）
        self.fc = nn.Linear(512, 5)

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

        # x = x.flatten(1)
        # -------------------------
        # 3. classifier（关键：保持时间维）
        # -------------------------
        #x = self.dropout(x)
        x = self.fc(x)  # (B, W, C)

        return x






class deepSleepNet_Encoder(nn.Module):
    def __init__(self,num_classes=5):
        super().__init__()


        # CNN
        self.cnn = deepsleepnet_Encoder(0.5)

        #self.dropout = nn.Dropout(0.5)


        # FC（softmax_linear）
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        x: (B, W, 1, 1)
        return: (B, W, num_classes)
        """

        # -------------------------
        # 1. CNN feature extraction
        # -------------------------
        x = self.cnn(x)  # (B, W, F)

        # x = x.flatten(1)
        # -------------------------
        # 3. classifier（关键：保持时间维）
        # -------------------------
        #x = self.dropout(x)
        x = self.fc(x)  # (B, W, C)

        return x









class DeepSleepNetEncode(nn.Module):  # current one!
    def __init__(self):
        super(DeepSleepNetEncode, self).__init__()
        self.lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True, dropout=0.5, num_layers=2)
        self.res = nn.Linear(512, 1024)
        self.linear = nn.Linear(1024, 512)

    def forward(self, x):
        # batch, 20, 128
        res = self.res(x)
        x, _ = self.lstm(x)
        x = x + res
        x = self.linear(x)
        return x







class deepSleepNet(nn.Module):
    def __init__(self,):
        super().__init__()


        # CNN
        self.cnn = deepsleepnet_Encoder(0.5)

        self.dropout = nn.Dropout(0.5)

        self.lstm=DeepSleepNetEncode()

        # FC（softmax_linear）
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        """
        x: (B, W, 1, 1)
        return: (B, W, num_classes)
        """

        # -------------------------
        # 1. CNN feature extraction
        # -------------------------
        x = self.cnn(x)  # (B, W, F)

        x = self.dropout(x)
        x = self.lstm(x)  # 时序平滑，输出: [B, W, 128]

        # x = x.flatten(1)
        # -------------------------
        # 3. classifier（关键：保持时间维）
        # -------------------------
        x = self.dropout(x)

        x = self.fc(x)  # (B, W, C)

        return x


