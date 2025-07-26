# modify from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_act.py # noqa:E501

import os
import torch
from torch import nn
from torch.autograd import Function
from functools import lru_cache
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch
from torch import Tensor

from torch.amp import custom_fwd

_MODULE_PATH = os.path.dirname(__file__)
build_dir = os.path.join(_MODULE_PATH, "torch_ext_build")
os.makedirs(build_dir, exist_ok=True)


@lru_cache(maxsize=1)
def get_fused_act_ext():
    return load(
        name="fused",
        sources=[
            os.path.join(_MODULE_PATH, "src", "fused_bias_act.cpp"),
            os.path.join(_MODULE_PATH, "src", "fused_bias_act_kernel.cu"),
        ],
        build_directory=build_dir,
        verbose=False,
    )


fused_act_ext = get_fused_act_ext()


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused_act_ext.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        (out,) = ctx.saved_tensors
        gradgrad_out = fused_act_ext.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused_act_ext.fused_bias_act(
            input, bias, empty, 3, 0, negative_slope, scale
        )
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

@custom_fwd(device_type="cuda", cast_inputs=torch.float)
def fused_leaky_relu(
    input: Tensor, bias: Tensor, negative_slope: float = 0.2, scale: float = 2**0.5
) -> Tensor:
    if input.device.type == "cpu":
        return scale * F.leaky_relu(
            input + bias.view((1, -1) + (1,) * (input.ndim - 2)),
            negative_slope=negative_slope,
        )
    else:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
