import torch
import torch.nn as nn
from .pointwise import PointWiseTN  # Assuming PointWiseTN is already migrated to PyTorch

class Encoder(nn.Module):
    def __init__(self, spatial_channels, temporal_channels_list, spatial_kernel_size, padding='same', drop=0.1, activation='relu', use_bias=True, use_bn=True):
        super(Encoder, self).__init__()

        assert len(spatial_kernel_size) == 3
        assert len(spatial_channels) == len(temporal_channels_list)

        self.contracting = nn.ModuleList()

        prev_s_c = 3
        for s_channels, t_channels in zip(spatial_channels, temporal_channels_list):
            spatial_conv = nn.Sequential()
            sc_in, sc_out = s_channels

            spatial_conv.append(nn.Conv3d(sc_in, sc_out, spatial_kernel_size, padding=padding, bias=use_bias))
            spatial_conv.append(nn.ReLU())

            bn = nn.BatchNorm3d(s_channels[-1]) if use_bn else nn.Identity()
            pool = nn.MaxPool3d((1, 2, 2))
            temporal_conv = PointWiseTN(
                channels=t_channels
            )
            self.contracting.append(nn.ModuleList([spatial_conv, bn, pool, temporal_conv]))

    def forward(self, x):
        ret = []
        for conv_s, bn, pool, conv_t in self.contracting:
            x = conv_s(x)
            s = x.shape

            # Add padding if necessary
            if s[-2] % 2:
                x = nn.functional.pad(x, (0, 0, 0, 1))

            if s[-1] % 2:
                x = nn.functional.pad(x, (0, 1))

            x = bn(x)
            x = pool(x)

            x = conv_t(x)
            ret.append(x)
        return ret