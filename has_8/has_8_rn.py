from typing import List, Optional, Type, Union
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, surrogate, neuron, functional
from has_8.has_8_base import SpikingEncodeDecodeBase

def sew_add_function(x: torch.Tensor, y: torch.Tensor):
    return x + y

def spiking_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def spiking_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class AnalogBasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            groups: int = 1,
            base_channels: int = 64,
            downsample: Optional[nn.Module] = None,
            expansion = 1) -> None:
        super().__init__()
        self.groups = groups
        self.base_channels = base_channels
        self.downsample = downsample
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out
    
class SpikingBasicBlock(SpikingEncodeDecodeBase):
    expansion = 1
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                stride: int,
                groups: int = 1,
                base_channels: int = 64,
                downsample: Optional[nn.Module] = None,
                expansion = 1):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.groups = groups
        self.base_channels = base_channels
        self.expansion = expansion
        self.conv1 = spiking_conv3x3(in_channels, out_channels, stride)
        self.bn1 = layer.BatchNorm2d(out_channels)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.conv2 = spiking_conv3x3(out_channels, out_channels)
        self.bn2 = layer.BatchNorm2d(out_channels)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.stride = stride
        functional.set_step_mode(self, step_mode='m')

    def _spiking_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_add_function(identity, out)

        return out
    
class FusionBasicBlock(nn.Module):
    expansion = 1
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 groups: int = 1,
                 base_channels: int = 64,
                 ann_downsample: Optional[nn.Module] = None,
                 snn_downsample: Optional[nn.Module] = None):
        super().__init__()
        self.groups = groups
        self.base_channels = base_channels
        self.snn_basic_block = SpikingBasicBlock(in_channels, 
                                                 out_channels, 
                                                 stride,
                                                 groups,
                                                 base_channels, 
                                                 snn_downsample,
                                                 self.expansion)
        
        self.ann_basic_block = AnalogBasicBlock(in_channels,
                                                out_channels,
                                                stride,
                                                groups,
                                                base_channels,
                                                ann_downsample,
                                                self.expansion)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.snn_basic_block(x) + self.ann_basic_block(x)
        return x

class AnalogBottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 64,
            expansion: int = 4,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels
        self.expansion = expansion

        channels = int(out_channels * (base_channels / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (stride, stride), (1, 1), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, int(out_channels * self.expansion), (1, 1), (1, 1), (0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_channels * self.expansion))
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out

class SpikingBottleneck(SpikingEncodeDecodeBase):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 stride: int, 
                 downsample=None, 
                 groups=1,
                 base_channels: int = 64,
                 expansion = 4):
        super().__init__()
        self.expansion = expansion
        width = int(out_channels * (base_channels / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = spiking_conv1x1(in_channels, width)
        self.bn1 = layer.BatchNorm2d(width)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.conv2 = spiking_conv3x3(width, width, stride, groups)
        self.bn2 = layer.BatchNorm2d(width)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.conv3 = spiking_conv1x1(width, out_channels * self.expansion)
        self.bn3 = layer.BatchNorm2d(out_channels * self.expansion)
        self.sn3 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.stride = stride
        functional.set_step_mode(self, step_mode='m')

    def _spiking_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sn3(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_add_function(out, identity)

        return out
    
class FusionBottleneck(nn.Module):
    expansion = 4
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int, 
                 groups=1,
                 base_channels: int = 64,
                 ann_downsample: Optional[nn.Module] = None,
                 snn_downsample: Optional[nn.Module] = None):
        super().__init__()
        self.snn_bottleneck = SpikingBottleneck(in_channels, 
                                                 out_channels,
                                                 stride, 
                                                 groups=groups,
                                                 base_channels=base_channels,
                                                 downsample=snn_downsample,
                                                 expansion=self.expansion)
        
        self.ann_bottleneck = AnalogBottleneck(in_channels,
                                               out_channels,
                                               stride,
                                               groups=groups,
                                               base_channels=base_channels,
                                               downsample=ann_downsample,
                                               expansion=self.expansion)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.snn_bottleneck(x) + self.ann_bottleneck(x)
        return x
    
class AnalogInitConv(nn.Module):
    def __init__(
            self,
            in_channels = 3,
            hidden_channels = 64) -> None:
        super().__init__()
       
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, (7, 7), (2, 2), (3, 3), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2), (1, 1))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.init_conv(x)

class SpikingInitConv(SpikingEncodeDecodeBase):
    def __init__(self,
                 in_channels = 3,
                 hidden_channels = 64):
        super().__init__()
        self.init_conv = nn.Sequential(
            layer.Conv2d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3, bias=False),
            layer.BatchNorm2d(hidden_channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        functional.set_step_mode(self, step_mode='m')

    def _spiking_forward(self, x):
        return self.init_conv(x)
    
class FusionInitConv(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 hidden_channels = 64):
        super().__init__()
        self.snn_init_conv = SpikingInitConv(in_channels, hidden_channels)
        
        self.ann_init_conv = AnalogInitConv(in_channels, hidden_channels)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.snn_init_conv(x) + self.ann_init_conv(x)
        return x
    
class SpikingLinear(SpikingEncodeDecodeBase):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.sl = nn.Sequential(
            layer.Linear(in_channels, out_channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self, step_mode='m')

    def _spiking_forward(self, 
                        x: torch.Tensor) -> torch.Tensor:
        return self.sl(x)
    
class AnalogLinear(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.al = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        return self.al(x)
    
class FusionLinear(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.snn_linear = SpikingLinear(in_channels, out_channels)
        self.ann_linear = AnalogLinear(in_channels, out_channels)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.snn_linear(x) + self.ann_linear(x)
        return x

class HAS8ResNet(nn.Module):
    def __init__(
            self,
            arch_cfg: List[int],
            block: Type[Union[FusionBasicBlock, FusionBottleneck]],
            input_channels: int = 3,
            groups: int = 1,
            channels_per_group: int = 32,
            num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64
        self.dilation = 1
        self.input_channels = input_channels
        self.groups = groups
        self.base_channels = channels_per_group

        self.init_conv = FusionInitConv(in_channels=input_channels)

        self.layer1 = self._make_layer(arch_cfg[0], block, self.base_channels, 1)
        self.layer2 = self._make_layer(arch_cfg[1], block, self.base_channels*2, 2)
        self.layer3 = self._make_layer(arch_cfg[2], block, self.base_channels*(2**2), 2)
        self.layer4 = self._make_layer(arch_cfg[3], block, self.base_channels*(2**3), 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.base_channels*(2**3)*block.expansion, num_classes)

        # Initialize neural network weights
        # self._initialize_weights()

    def _make_layer(
            self,
            repeat_times: int,
            block: Type[Union[FusionBasicBlock, FusionBottleneck]],
            channels: int,
            stride: int = 1,
    ) -> nn.Sequential:
        ann_downsample = None
        snn_downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:
            ann_downsample = nn.Sequential(
                                nn.Conv2d(self.in_channels, channels * block.expansion, (1, 1), (stride, stride), (0, 0), bias=False),
                                nn.BatchNorm2d(channels * block.expansion),
                            )
            snn_downsample = nn.Sequential(
                                layer.Conv2d(self.in_channels, channels * block.expansion, (1, 1), (stride, stride), (0, 0), bias=False),
                                layer.BatchNorm2d(channels * block.expansion),
                            )

        layers = [
            block(
                self.in_channels,
                channels,
                stride,
                self.groups,
                self.base_channels,
                ann_downsample,
                snn_downsample
            )
        ]
        self.in_channels = channels * block.expansion
        for _ in range(1, repeat_times):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    1,
                    self.groups,
                    self.base_channels
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward_impl(x)
        return out

    # Idk if this support torch.script function but probably not
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, layer.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

def has8_rn_b32_m2_d4(input_channels, num_classes, **kwargs):
    model = HAS8ResNet([2, 2, 2, 2], 
                       FusionBasicBlock,
                       input_channels=input_channels, 
                       num_classes=num_classes, 
                       channels_per_group=32, 
                       **kwargs)
    return model

def has8_rn_b64_m2_d4(input_channels, num_classes, **kwargs):
    model = HAS8ResNet([2, 2, 2, 2], 
                       FusionBasicBlock,
                       input_channels=input_channels, 
                       num_classes=num_classes,  
                       channels_per_group=64, 
                       **kwargs)
    return model

#Untested experimental models
def has8_rn34(**kwargs):
    model = HAS8ResNet([3, 4, 6, 3], 
                       FusionBasicBlock, 
                       **kwargs)
    return model

def has8_rn50(**kwargs):
    model = HAS8ResNet([3, 4, 6, 3], 
                       FusionBottleneck, 
                       **kwargs)
    return model

def has8_rn101(**kwargs):
    model = HAS8ResNet([3, 4, 23, 3], 
                       FusionBottleneck, 
                       **kwargs)
    return model

def has8_rn152(**kwargs):
    model = HAS8ResNet([3, 8, 36, 3], 
                       FusionBottleneck, 
                       **kwargs)
    return model