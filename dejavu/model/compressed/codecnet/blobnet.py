import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .preprocessing import Preprocessing  # Assuming this is also migrated to PyTorch

class CodecNet(nn.Module):
    def __init__(
            self,
            input_shape,
            e_spatial_channels,
            e_temporal_channels_list,
            e_spatial_kernel_size,
            e_activation,
            e_use_bn,
            d_spatial_channels,
            d_spatial_kernel_size):
        super(CodecNet, self).__init__()
        
        self.preprocessing = Preprocessing(channels=input_shape[0])
        self.encoder = Encoder(
            spatial_channels=e_spatial_channels,
            temporal_channels_list=e_temporal_channels_list,
            spatial_kernel_size=e_spatial_kernel_size,
            padding="same",
            activation=e_activation,
            use_bias=True,
            use_bn=e_use_bn
        )

        dummy_input = torch.zeros(input_shape).unsqueeze(0)
        dummy_encoder_output = self.encoder(dummy_input)
        x_rev = [x[:, :, :1] for x in reversed(dummy_encoder_output)]
        shapes = [x.shape for x in x_rev]
        shapes.append(input_shape)

        self.decoder = Decoder(
            input_shapes=shapes,
            spatial_channels=d_spatial_channels,
            spatial_kernel_size=d_spatial_kernel_size,
        )
        self.input_shape = input_shape

    def forward(self, x):
        x = self.preprocessing(x)
        x_enc = self.encoder(x)
        x_rev = [x[:, :, :1] for x in reversed(x_enc)]
        x_dec = self.decoder(x_rev)
        out = x_dec.squeeze(2)

        return out

# Example usage
if __name__ == '__main__':
    model = CodecNet(
        input_shape=[3, 30, 16, 16],
        # Encoder
        e_spatial_channels=[(3, 16), (16, 32), (32, 64), (64, 128)],
        e_spatial_kernel_size=[1, 2, 2],
        e_temporal_channels_list=[[30, 30], [30, 30], [30, 30], [30, 30]],
        e_activation='relu',
        e_use_bn=True,
        # Decoder
        d_spatial_channels=[(128, 64), (128, 32), (64, 16), (32, 16)],
        d_spatial_kernel_size=[1, 2, 2],
    )

    dummy_input = torch.zeros([1, 3, 30, 16, 16])

    dummy_output = model(dummy_input)
    assert dummy_output.shape == (1, 1, 16, 16)

