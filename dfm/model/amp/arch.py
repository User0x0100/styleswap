from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F


EPS = 1e-6


class Downscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=2)

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(self.conv1(x), 0.1)


class Upscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pixel_shuffle(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")

    def forward(self, inp: Tensor) -> Tensor:
        x = self.conv1(inp)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(inp + x, 0.2)
        return x


class Encoder(nn.Module):
    def __init__(self, input_ch: int, e_dims: int, ae_dims: int, resolution: int) -> None:
        super().__init__()
        self.down1 = Downscale(input_ch, e_dims, kernel_size=5)
        self.res1 = ResidualBlock(e_dims)
        self.down2 = Downscale(e_dims, e_dims * 2, kernel_size=5)
        self.down3 = Downscale(e_dims * 2, e_dims * 4, kernel_size=5)
        self.down4 = Downscale(e_dims * 4, e_dims * 8, kernel_size=5)
        self.down5 = Downscale(e_dims * 8, e_dims * 8, kernel_size=5)
        self.res5 = ResidualBlock(e_dims * 8)
        self.dense1 = nn.Linear(((resolution // (2**5)) ** 2) * e_dims * 8, ae_dims)
        self.flatten = nn.Flatten()

    @staticmethod
    def pixel_norm(x: Tensor) -> Tensor:
        return x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + EPS)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.res5(x)

        x = self.pixel_norm(self.flatten(x))
        x = self.dense1(x)
        return x


class Inter(nn.Module):
    def __init__(self, ae_dims: int, inter_res: int, inter_dims: int) -> None:
        super().__init__()

        self.dense2 = nn.Linear(ae_dims, inter_res * inter_res * inter_dims)
        self.inter_res = inter_res
        self.inter_dims = inter_dims

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense2(x)
        x = x.view(-1, self.inter_dims, self.inter_res, self.inter_res)
        return x


class Decoder(nn.Module):
    def __init__(self, inter_dims: int, d_dims: int, d_mask_dims: int) -> None:
        super().__init__()

        self.upscale0 = Upscale(inter_dims, d_dims * 8, kernel_size=3)
        self.upscale1 = Upscale(d_dims * 8, d_dims * 8, kernel_size=3)
        self.upscale2 = Upscale(d_dims * 8, d_dims * 4, kernel_size=3)
        self.upscale3 = Upscale(d_dims * 4, d_dims * 2, kernel_size=3)

        self.res0 = ResidualBlock(d_dims * 8, kernel_size=3)
        self.res1 = ResidualBlock(d_dims * 8, kernel_size=3)
        self.res2 = ResidualBlock(d_dims * 4, kernel_size=3)
        self.res3 = ResidualBlock(d_dims * 2, kernel_size=3)

        self.upscalem0 = Upscale(inter_dims, d_mask_dims * 8, kernel_size=3)
        self.upscalem1 = Upscale(d_mask_dims * 8, d_mask_dims * 8, kernel_size=3)
        self.upscalem2 = Upscale(d_mask_dims * 8, d_mask_dims * 4, kernel_size=3)
        self.upscalem3 = Upscale(d_mask_dims * 4, d_mask_dims * 2, kernel_size=3)
        self.upscalem4 = Upscale(d_mask_dims * 2, d_mask_dims * 1, kernel_size=3)
        self.out_convm = nn.Conv2d(d_mask_dims * 1, 1, kernel_size=1, padding="same")

        self.out_conv = nn.Conv2d(d_dims * 2, 3, kernel_size=1, padding="same")
        self.out_conv1 = nn.Conv2d(d_dims * 2, 3, kernel_size=3, padding="same")
        self.out_conv2 = nn.Conv2d(d_dims * 2, 3, kernel_size=3, padding="same")
        self.out_conv3 = nn.Conv2d(d_dims * 2, 3, kernel_size=3, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:

        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)
        x = self.upscale3(x)
        x = self.res3(x)

        x0 = self.out_conv(x)
        x1 = self.out_conv1(x)
        x2 = self.out_conv2(x)
        x3 = self.out_conv3(x)

        x_concat = torch.cat([x0, x1, x2, x3], dim=1)

        x = torch.sigmoid(self.pixel_shuffle(x_concat))

        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        m = self.upscalem3(m)
        m = self.upscalem4(m)
        m = torch.sigmoid(self.out_convm(m))

        return x, m


class AMPModel(nn.Module):
    def __init__(self, resolution: int = 320, input_ch=3, e_dims: int = 64, ae_dims: int = 256, inter_dims: int = 1536, d_dims: int = 64, d_mask_dims: int = 22) -> None:
        super().__init__()

        inter_res = resolution // 32

        self.encoder = Encoder(input_ch=input_ch, e_dims=e_dims, ae_dims=ae_dims, resolution=resolution)
        self.inter_src = Inter(ae_dims=ae_dims, inter_res=inter_res, inter_dims=inter_dims)
        self.inter_dst = Inter(ae_dims=ae_dims, inter_res=inter_res, inter_dims=inter_dims)
        self.decoder = Decoder(inter_dims=inter_dims, d_dims=d_dims, d_mask_dims=d_mask_dims)

    def forward(self, x: Tensor, morph_value: float) -> Tuple[Tensor, Tensor]:

        gpu_dst_code = self.encoder(x)

        gpu_dst_inter_src_code = self.inter_src(gpu_dst_code)
        gpu_dst_inter_dst_code = self.inter_dst(gpu_dst_code)

        inter_dims_slice = int(self.inter_src.inter_dims * morph_value)

        gpu_src_dst_code = torch.cat([gpu_dst_inter_src_code[:, :inter_dims_slice, :, :], gpu_dst_inter_dst_code[:, inter_dims_slice:, :, :]], dim=1)

        x, m = self.decoder(gpu_src_dst_code)

        return x, m


if __name__ == "__main__":
    m = AMPModel()
    x = torch.randn(8, 3, 320, 320)
    x, m = m(x, 0.5)
    print(x.size(), m.size())
