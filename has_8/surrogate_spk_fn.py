import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch import nn
import math

@torch.jit.script
def rate_decode(x: torch.Tensor):
    return x.mean(0)

@torch.jit.script
def bpd_decode(x: torch.Tensor):
    device = x.device
    dtype = x.dtype
    weights = torch.tensor([2 ** i for i in reversed(range(8))], device=device, dtype=dtype)
    shape = [8] + [1] * (x.ndim - 1)
    weights = weights.view(*shape)
    decoded = (x * weights).sum(dim=0)
    return decoded / 255.0

@torch.jit.script
def _optional_rescale(tensor: torch.Tensor):
    if torch.all(tensor >= 0) and torch.all(tensor <= 1):
        return tensor
    else:
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:
            return torch.ones_like(tensor)
        return (tensor - min_val) / (max_val - min_val)
    
@torch.jit.script    
def grad_rescale(x: torch.Tensor):
    k = torch.tensor([n for n in range(8)], device=x.device).view(-1, 1, 1, 1, 1)
    denom = torch.tensor([2**(7-n) for n in range(8)], device=x.device).view(-1, 1, 1, 1, 1)
    return (k*x)/denom

@torch.jit.script
def bit_plane_coding(x: torch.Tensor,  
                     bit_per_channels: int):
    res = []
    for _ in range(bit_per_channels):
        res.append(torch.remainder(x, 2))
        x = torch.div(x, 2, rounding_mode="floor")
    res = torch.stack(res)
    return res
    
class SurrogateFunctionBase(Function):
    generate_vmap_rule = True
    @staticmethod
    def forward(x, max, alpha):
        bit_per_channels = math.ceil(math.log2(max))
        return bit_plane_coding(x, bit_per_channels)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, max, alpha = inputs
        ctx.save_for_backward(x, torch.tensor(max), torch.tensor(alpha))
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Gradient is not implemented.")
    
class SigSineBase(SurrogateFunctionBase):
    @staticmethod
    def backward(ctx, grad_output):
        x, max, alpha = ctx.saved_tensors
        max = max.item()
        alpha = alpha.item()
        n_bits = math.ceil(math.log2(max))
        c = torch.tensor([2**i for i in range(n_bits)], device=x.device)
        u = alpha*torch.sin((x*torch.pi)/c.view(-1, 1, 1, 1, 1))
        du = (alpha*torch.pi)/c.view(-1, 1, 1, 1, 1)*torch.cos((x*torch.pi)/c.view(-1, 1, 1, 1, 1))
        grad_input = grad_output*grad_rescale(F.sigmoid(u)*(1 - F.sigmoid(u))*du)
        return grad_input, None, None  # No gradient for max

class SigSine(nn.Module):
    def __init__(self,
                 max:float=255.,
                 alpha:int=-10):
        super().__init__()
        self.max = max
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _optional_rescale(x)*self.max
        return SigSineBase.apply(x, self.max, self.alpha)

class TanhSineBase(SurrogateFunctionBase):
    @staticmethod
    def backward(ctx, grad_output):
        x, max, alpha = ctx.saved_tensors
        max = max.item()
        alpha = alpha.item()
        n_bits = math.ceil(math.log2(max))
        c = torch.tensor([2**i for i in range(n_bits)], device=x.device)
        u = alpha*torch.sin((x*torch.pi)/c.view(-1, 1, 1, 1, 1))
        du = (alpha*torch.pi)/c.view(-1, 1, 1, 1, 1)*torch.cos((x*torch.pi)/c.view(-1, 1, 1, 1, 1))
        grad_input = grad_output*grad_rescale((1/2)*((1/torch.cosh(u))**2)*du)
        return grad_input, None, None  # No gradient for max

class TanhSine(nn.Module):
    def __init__(self,
                 max:float=255.,
                 alpha:int=-10):
        super().__init__()
        self.max = max
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _optional_rescale(x)*self.max
        return TanhSineBase.apply(x, self.max, self.alpha)
    

class FourierSineBase(SurrogateFunctionBase):
    @staticmethod
    def backward(ctx, grad_output):
        x, max, alpha = ctx.saved_tensors
        max = max.item()
        alpha = alpha.item()
        n_bits = math.ceil(math.log2(max))
        c = torch.tensor([2**i for i in range(n_bits)], device=x.device)
        chain = torch.stack([(2/c.view(-1, 1, 1, 1, 1))*torch.cos((x*torch.pi*n)/c.view(-1, 1, 1, 1, 1)) for n in range(1, 1+alpha*2, 2)])
        chain = 1/2 - chain.sum(0)
        grad_input = grad_output*grad_rescale(chain)
        return grad_input, None, None  # No gradient for max

class FourierSine(nn.Module):
    def __init__(self,
                 max:float=255.,
                 alpha:int=5):
        super().__init__()
        self.max = max
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _optional_rescale(x)*self.max
        return FourierSineBase.apply(x, self.max, self.alpha)