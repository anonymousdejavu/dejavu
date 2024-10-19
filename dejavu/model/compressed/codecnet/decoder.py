import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            desired_shape,
            use_bias=True,
            use_bn=True,
            use_dropout=False,
            use_relu=False
        ):
        super(UpsampleBlock, self).__init__()

        self.desired_shape = desired_shape
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=(1, 2, 2), padding=0, bias=use_bias)
        self.use_dropout = use_dropout
        self.use_relu = use_relu

    def forward(self, x):
        if self.use_relu:
            x = F.relu(x)
        if self.use_dropout:
            x = F.dropout(x, 0.2)

        x = self.conv(x)
        s = x.shape

        # Crop if necessary
        if s[-2] > self.desired_shape[-2]:
            x = x[..., :self.desired_shape[-2], :]

        if s[-1] > self.desired_shape[-1]:
            x = x[..., :self.desired_shape[-1]]

        return x

class Decoder(nn.Module):
    def __init__(self, input_shapes, spatial_channels, spatial_kernel_size, use_relu=True, use_dropout=True):
        super(Decoder, self).__init__()

        assert len(input_shapes) == len(spatial_channels) + 1
        self.expanding = nn.ModuleList()

        for i, spatial_channel in enumerate(spatial_channels):
            sc_in, sc_out = spatial_channel
            up = UpsampleBlock(
                sc_in,
                sc_out,
                spatial_kernel_size,
                input_shapes[i + 1],
                use_relu=use_relu,
                use_dropout=use_dropout
            )
            bn = nn.BatchNorm3d(sc_out) if i < len(spatial_channels) - 1 else None
            self.expanding.append(nn.Sequential(up, bn))

        self.final = nn.Conv3d(sc_out, 1, 1)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        x = inputs[0]
        for inp, (up, bn) in zip(inputs[1:], self.expanding):
            x = up(x)
            if bn is not None:
                x = bn(x)
            x = torch.cat([x, inp], dim=1)

        x = self.expanding[-1][0](x)
        x = self.final(x)
        out = self.activation(x)

        return out
