import torch
import torch.nn as nn
from has_8.surrogate_spk_fn import FourierSine, bpd_decode

class SpikingEncodeDecodeBase(nn.Module):
    def __init__(self,
                 encode_block: nn.Module = FourierSine(),
                 decode_fn = bpd_decode):
        super().__init__()
        self.encoder_block = encode_block
        self.decode_fn = decode_fn

    def _spiking_encode(self, 
                       x: torch.Tensor) -> torch.Tensor:
        return self.encoder_block(x)

    def _spiking_forward(self, 
                        x: torch.Tensor) -> torch.Tensor:
        return NotImplementedError("Forward for spiking block is not implemented.")
    
    def _spiking_decode(self, 
                       x: torch.Tensor) -> torch.Tensor:
        self.decode_fn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._spiking_encode(x)
        x = self._spiking_forward(x)
        return self._spiking_decode(x)
