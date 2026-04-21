import math
from functools import partial
from math import log2
from typing import Tuple

import torch
from torch import Tensor
from linear_attention_transformer import ImageLinearAttention
from torch import nn
from torch.autograd import grad


def gradient_penalty(images: Tensor, outputs: Tensor, weight=10) -> Tensor:
    batch_size = images.shape[0]
    gradients = grad(outputs=outputs, inputs=images, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles: Tensor, images: Tensor) -> Tensor:
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = grad(outputs=outputs, inputs=styles, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return (pl_grads**2).sum(dim=2).mean(dim=1).sqrt()


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Flatten(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.index)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) * self.g


attn_and_ff = lambda chan: nn.Sequential(
    *[
        Residual(Rezero(ImageLinearAttention(chan, norm_queries=True))),
        Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(chan * 2, chan, 1)))),
    ]
)


def double_conv(chan_in: int, chan_out: int):
    return nn.Sequential(nn.Conv2d(chan_in, chan_out, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(chan_out, chan_out, 3, padding=1), nn.LeakyReLU(0.2))


class DownBlock(nn.Module):
    def __init__(self, input_channels: int, filters: int, downsample: bool = True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, filters: int):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride=2)
        self.net = double_conv(input_channels, filters)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size=(h * 2, w * 2))
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x


class UNetDiscriminator(nn.Module):
    def __init__(self, image_size: int, network_capacity: int = 16, transparent: bool = False, fmap_max: int = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 3)
        num_init_filters = 3 if not transparent else 4

        filters = [num_init_filters] + [(network_capacity) * (2**i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):

            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample=is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(nn.LeakyReLU(0.2), nn.AvgPool2d(image_size // (2**num_layers)), Flatten(1), nn.Linear(last_chan, 1))

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(3, 1, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        residuals = []

        for down_block, attn_block in zip(self.down_blocks, self.attn_blocks):
            x, unet_res = down_block(x)
            residuals.append(unet_res)

            if attn_block is not None:
                x = attn_block(x)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        for up_block, res in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return enc_out.squeeze(), dec_out


if __name__ == "__main__":

    ud = UNetDiscriminator(image_size=512)
