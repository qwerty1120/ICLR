import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import math
from .resnet import ResidualDenseNet


class GafToProbRegressor(nn.Module):
    """GAF 이미지 → n_timestamps 개 캔들 발생 확률 (로짓)"""
    def __init__(self,
                 in_channels: int,
                 n_timestamps: int,
                 encode_channels: list[int],
                 encode_block_type: str = "DenseNet",
                 encode_block_dim: int = 3,
                 image_size: int = 32):
        super().__init__()
        self.n_timestamps = n_timestamps
        self.encode_channels = encode_channels
        self.encode_block_type = encode_block_type.lower()
        self.encode_block_dim = encode_block_dim
        self.image_size = image_size

        if self.encode_block_type != "densenet":
            raise ValueError("Only 'DenseNet' encoder is implemented in this example.")

        self.encoder = self._build_encoder(in_channels)
        self.fc      = nn.Linear(encode_channels[-1], n_timestamps)  # 로짓

    # ----------------------------------------------------
    def _build_encoder(self, in_channels: int) -> nn.Module:
        layers = [
            nn.Conv2d(in_channels, self.encode_channels[0], kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ]

        for idx, (cin, cout) in enumerate(zip(self.encode_channels[:-1], self.encode_channels[1:])):
            layers += [
                ResidualDenseNet(channels=cin, depth=self.encode_block_dim),
                nn.Conv2d(in_channels=(self.encode_block_dim+1)*cin, out_channels=cout, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            ]
        return nn.Sequential(*layers)

    # ----------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        feat = self.encoder(x)                 # (B, C_out, H', W')
        feat = F.adaptive_avg_pool2d(feat, 1)  # (B, C_out, 1, 1)
        feat = feat.flatten(1)                 # (B, C_out)
        logits = self.fc(feat)                 # (B, n_timestamps)
        return logits

class GafToGafRegressor(nn.Module):
    
    def __init__(self, in_features, out_features, encode_channels, decode_channels,
                 encode_block_type, encode_block_dim, image_size):
        
        '''
        encode_block_dim is the depth for ResidualDenseNet and the width for ResNeXt
        '''
        
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.encode_channels = encode_channels
        self.decode_channels = decode_channels
        self.encode_block_type = encode_block_type
        self.encode_block_dim = encode_block_dim
        self.image_size = image_size

        self.net = self.build_net()
        
    def compute_reduction(self):
        
        size = self.image_size
        size = math.floor(1*(size - 5)+1) # Initial Convolution
        size = math.floor(0.5*(size - 2)+1) # MaxPool2d
        for _ in range(len(self.encode_channels)-1):
            size = math.floor(0.5*(size - 3)+1) # Convolution
            size = math.floor(0.5*(size - 2)+1) # MaxPool2d

        return size
    
    def compute_resizing(self):
        
        size = self.image_size
        size = (size-4)/2+1
        for _ in self.decode_channels:
            size = (size-4)/2+1
            
        return math.ceil(size / self.compute_reduction()*1000)/1000
    
    def build_net(self):
        
        if self.encode_block_type.lower() == 'densenet':
            return self.build_residual_dense_net()
        else:
            raise ValueError(f'Unknown encode block type')
            
    def build_residual_dense_net(self):
        
        '''
        Source:
        https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
        '''
        
        encoder = nn.Sequential(nn.Conv2d(in_channels=self.in_features,
                                          out_channels=self.encode_channels[0],
                                          kernel_size=5,
                                          stride=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2))
        
        encoder_in = self.encode_channels[:-1]
        encoder_out = self.encode_channels[1:]
        print(f'encoder_in : {encoder_in}')
        print(f'encoder_out: {encoder_out}\n')

        for i in range(len(encoder_in)):
            print(f'Adding ResidualDenseNet({encoder_in[i]}, {self.encode_block_dim})')
            encoder = nn.Sequential(
                encoder,
                ResidualDenseNet(channels=encoder_in[i], depth=self.encode_block_dim),
                nn.Conv2d(in_channels=(self.encode_block_dim+1)*encoder_in[i], 
                          out_channels=encoder_out[i],
                          kernel_size=3,
                          stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            print(f'Adding Conv2d({encoder_in[i]},{encoder_out[i]})')
        
        decoder_in = [encoder_out[-1]] + self.decode_channels
        decoder_out = self.decode_channels + [self.out_features]
        
        print(f'\ndecoder_in : {decoder_in}')
        print(f'decoder_out: {decoder_out}\n')
        
        decoder = nn.Sequential(nn.Upsample(scale_factor=self.compute_resizing()))
        print(f'resize factor: {self.compute_resizing()}\n')
        
        for i in range(len(decoder_in)):
            print(f'Adding ConvTranspose2d({decoder_in[i]}, {decoder_out[i]})')
            decoder = nn.Sequential(
                decoder,
                nn.ConvTranspose2d(in_channels=decoder_in[i],
                                out_channels=decoder_out[i],
                                kernel_size=4,
                                stride=2)
            )
            if i < len(decoder_in)-1:
                decoder = nn.Sequential(
                    decoder,
                    nn.ReLU()
                )
        
        return nn.Sequential(encoder, decoder)
    
    def forward(self, x):
        
        return self.net(x)