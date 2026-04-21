# modify from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_act.py # noqa:E501

from pathlib import Path
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd.function import FunctionCtx
from torch import Tensor, nn
from torch.amp import custom_fwd
from torch.autograd import Function
from torch.utils.cpp_extension import load

from ... import mode


BASE_DIR = Path(__file__).resolve().parent

fused_act_ext = load(name="fused", sources=[BASE_DIR / "src" / "fused_bias_act.cpp", BASE_DIR / "src" / "fused_bias_act_kernel.cu"], verbose=False)


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx: FunctionCtx, grad_output: Tensor, out: Tensor, negative_slope: float, scale: float) -> tuple[Tensor, Tensor]:

        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty: Tensor = grad_output.new_empty(0)

        grad_input: Tensor = fused_act_ext.fused_bias_act(grad_output, empty, out, 3, 1, negative_slope, scale)

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx: FunctionCtx, gradgrad_input: Tensor, gradgrad_bias: Tensor) -> tuple[Tensor, None, None, None]:
        (out,) = ctx.saved_tensors
        gradgrad_out = fused_act_ext.fused_bias_act(gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale)

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, bias: Tensor, negative_slope: float, scale: float) -> Tensor:
        empty = input.new_empty(0)
        out = fused_act_ext.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(grad_output, out, ctx.negative_slope, ctx.scale)

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel: int, negative_slope: float = 0.2, scale: float = 2**0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


@custom_fwd(device_type="cuda", cast_inputs=torch.float)
def fused_leaky_relu(input: Tensor, bias: Tensor, negative_slope: float = 0.2, scale: float = 2**0.5) -> Tensor:

    if input.device.type == "cpu" or mode.EXPORT_MODE:
        return scale * F.leaky_relu(input + bias.view((1, -1) + (1,) * (input.ndim - 2)), negative_slope=negative_slope)
    else:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
