import torch
import torch.nn as nn

class PointWiseTN(nn.Module):
    def __init__(self, channels, drop=0.2):
        super(PointWiseTN, self).__init__()

        layers = []
        for c in channels:
            layers.append(nn.Conv1d(c, c, 1, bias=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))

        self.inner = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        # [N, C, T, H, W] ==> [N, C, H, W, T]
        N, C, T, H, W = x.shape
        residual = x
        x = x.reshape(N*C, T, H*W)
        x = self.inner(x)
        # [N, C, H, W, T] ==> [N, C, T, H, W]
        x = x.reshape(N, C, -1, H, W)

        x = x + residual
        x = self.relu(x)
        return x
