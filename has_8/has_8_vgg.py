import torch
from torch import nn
from has_8.has_8_base import SpikingEncodeDecodeBase
from spikingjelly.activation_based import layer, neuron, surrogate, functional
import math

class SPKConvBlock(SpikingEncodeDecodeBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_kernel_size=3,
                 pool_kernel_size=2,
                 pool_kernel_stride=2):
        super().__init__()
        self.sdv = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=1, bias=False),
            layer.BatchNorm2d(out_channels, momentum=math.sqrt(0.1)),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(pool_kernel_size, pool_kernel_stride)
        )
        functional.set_step_mode(self, step_mode='m')

    def _spiking_forward(self, 
                        x: torch.Tensor) -> torch.Tensor:
        return self.sdv(x)
    
class ANNConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_kernel_size=3,
                 pool_kernel_size=2,
                 pool_kernel_stride=2):
        super().__init__()
        self.adv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.AvgPool2d(pool_kernel_size, pool_kernel_stride),
        )

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        return self.adv(x)
    
class SPKLinearBlock(SpikingEncodeDecodeBase):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.sdv = nn.Sequential(
            layer.Linear(in_channels, 1024),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(1024, 512),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(512, out_channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self, step_mode='m')

    def _spiking_forward(self, 
                        x: torch.Tensor) -> torch.Tensor:
        return self.sdv(x)
    
class ANNLinearBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.adv = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        return self.adv(x)
    
class ConvFusionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_kernel_size=3,
                 pool_kernel_size=2,
                 pool_kernel_stride=2):
        super().__init__()
        self.sdv_block = SPKConvBlock(in_channels, 
                                  out_channels,
                                  conv_kernel_size,
                                  pool_kernel_size,
                                  pool_kernel_stride)
        self.adv_block = ANNConvBlock(in_channels, 
                                  out_channels,
                                  conv_kernel_size,
                                  pool_kernel_size,
                                  pool_kernel_stride)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.sdv_block(x) + self.adv_block(x)
        return x
    
class LinearFusionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.sdv_block = SPKLinearBlock(in_channels, 
                                        out_channels)
        self.adv_block = ANNLinearBlock(in_channels, 
                                        out_channels)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        return self.sdv_block(x) + self.adv_block(x)
    
class HAS8VGG(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1000,
                 multiplier=2,
                 depth=4,
                 internal_channel_count=16,
                 conv_kernel_size=3,
                 pool_kernel_size=2,
                 pool_kernel_stride=2):
        super().__init__()
        self.dv = nn.Sequential(
            ConvFusionBlock(in_channels,
                            in_channels*internal_channel_count,
                            conv_kernel_size,
                            pool_kernel_size,
                            pool_kernel_stride),

            *[ConvFusionBlock(in_channels*internal_channel_count*(multiplier**i),
                              in_channels*internal_channel_count*(multiplier**(i+1)),
                              conv_kernel_size,
                              pool_kernel_size,
                              pool_kernel_stride) for i in range(depth)],
        
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            LinearFusionBlock(in_channels*internal_channel_count*(multiplier**depth), 
                              out_channels)
        )

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.dv(x)
        return x
    
def has8_vgg_b16_m2_d4(in_channels, out_channels, **kwargs):
    model = HAS8VGG(in_channels=in_channels, 
                    out_channels=out_channels, 
                    internal_channel_count=16, 
                    **kwargs)
    return model

def has8_vgg_b24_m2_d4(in_channels, out_channels, **kwargs):
    model = HAS8VGG(in_channels=in_channels, 
                    out_channels=out_channels, 
                    internal_channel_count=24, 
                    **kwargs)
    return model