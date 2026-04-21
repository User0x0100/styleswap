from typing import Tuple
import warnings
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from dfm.model import EPS


class Downscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=5) -> None:
        """
        Downscale component.

        Args:
            in_ch (int): The number of input channels e.g. 3 for a standard RGB image
            out_ch (int): The number of output channels
            kernel_size (int): The kernel size used during the convolution operation
        """
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=kernel_size, stride=2, padding=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        return x

    def get_out_ch(self) -> int:
        return self.out_ch


class Encoder(nn.Module):
    def __init__(self, in_ch: int, e_ch: int) -> None:
        super().__init__()

        self.e_ch = e_ch

        self.down1 = Downscale(in_ch, e_ch, kernel_size=5)
        self.res1 = ResidualBlock(e_ch)

        self.down2 = Downscale(e_ch, e_ch * 2, kernel_size=5)
        self.down3 = Downscale(e_ch * 2, e_ch * 4, kernel_size=5)
        self.down4 = Downscale(e_ch * 4, e_ch * 8, kernel_size=5)
        self.down5 = Downscale(e_ch * 8, e_ch * 8, kernel_size=5)

        self.res5 = ResidualBlock(e_ch * 8)
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
        x = self.flatten(x)
        x = self.pixel_norm(x)
        return x

    def get_output_length(self, input_resolution: int) -> int:
        res = input_resolution // 32
        return (self.e_ch * 8) * res * res


class Upscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        """
        Deconvolution block used for upscaling the input tensor

        Args:
            in_ch (int): The number of input channels
            out_ch (int): The number of output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pixel_shuffle(x)
        return x


class Inter(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, lowest_dense_res) -> None:
        super().__init__()

        self.ae_out_ch = ae_out_ch
        self.lowest_dense_res = lowest_dense_res

        self.dense1 = nn.Linear(in_ch, ae_ch)
        self.dense2 = nn.Linear(ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense1(x)
        x = self.dense2(x)
        return x.view(x.shape[0], self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res)

    def get_out_ch(self) -> int:
        return self.ae_out_ch


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3) -> None:
        """
        Residule convolutional block

        Args:
            ch (int): The number of input and output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")

    def forward(self, inp: Tensor) -> Tensor:
        x = self.conv1(inp)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(inp + x, 0.2)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch: int, d_ch: int, d_mask_ch: int, o_ch: int = 3) -> None:
        """
        The docoder block

        Args:
            in_ch (int): The number of input channels
            d_ch (int): Parameter to control the number of output channels in each of the Upscale blocks
            d_mask_ch (int): Parameter to control the number of output channels in the mask Upscale block
            image_output_channels (int): The number of channels in the output image
        """
        super().__init__()

        self.upscale0 = Upscale(in_ch, d_ch * 8, kernel_size=3)
        self.upscale1 = Upscale(d_ch * 8, d_ch * 8, kernel_size=3)
        self.upscale2 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3)
        self.upscale3 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3)

        self.res0 = ResidualBlock(d_ch * 8, kernel_size=3)
        self.res1 = ResidualBlock(d_ch * 8, kernel_size=3)
        self.res2 = ResidualBlock(d_ch * 4, kernel_size=3)
        self.res3 = ResidualBlock(d_ch * 2, kernel_size=3)

        self.out_conv = nn.Conv2d(d_ch * 2, o_ch, kernel_size=1, padding="same")
        self.out_conv1 = nn.Conv2d(d_ch * 2, o_ch, kernel_size=3, padding="same")
        self.out_conv2 = nn.Conv2d(d_ch * 2, o_ch, kernel_size=3, padding="same")
        self.out_conv3 = nn.Conv2d(d_ch * 2, o_ch, kernel_size=3, padding="same")
        self.pixel_shuffle_out = nn.PixelShuffle(2)

        self.upscalem0 = Upscale(in_ch, d_mask_ch * 8, kernel_size=3)
        self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * 8, kernel_size=3)
        self.upscalem2 = Upscale(d_mask_ch * 8, d_mask_ch * 4, kernel_size=3)
        self.upscalem3 = Upscale(d_mask_ch * 4, d_mask_ch * 2, kernel_size=3)
        self.upscalem4 = Upscale(d_mask_ch * 2, d_mask_ch * 1, kernel_size=3)

        self.out_convm = nn.Conv2d(d_mask_ch * 1, 1, kernel_size=1, padding="same")

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:

        x = self.upscale0(inp)
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

        x = torch.sigmoid(self.pixel_shuffle_out(x_concat))

        m = self.upscalem0(inp)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        m = self.upscalem3(m)
        m = self.upscalem4(m)

        m = torch.sigmoid(self.out_convm(m))
        return x, m


class LIAEUDT(nn.Module):
    def __init__(self, resolution: int = 256, e_dims: int = 64, ae_dims: int = 256, d_dims: int = 64, d_mask_dims: int = 24) -> None:
        super().__init__()

        self.resolution = resolution
        self.e_dims = e_dims
        self.ae_dims = ae_dims
        self.d_dims = d_dims
        self.d_mask_dims = d_mask_dims

        self.encoder = Encoder(in_ch=3, e_ch=e_dims)
        encoder_out_ch = self.encoder.get_output_length(input_resolution=resolution)

        lowest_dense_res = resolution // 32
        self.inter_B = Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims * 2, lowest_dense_res=lowest_dense_res)
        self.inter_AB = Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims * 2, lowest_dense_res=lowest_dense_res)

        inter_AB_out_ch = self.inter_AB.get_out_ch()
        inter_B_out_ch = self.inter_B.get_out_ch()
        inters_out_ch = inter_AB_out_ch + inter_B_out_ch

        self.decoder = Decoder(in_ch=inters_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        code = self.encoder(x)
        inter_AB = self.inter_AB(code)
        inter_concat = torch.cat((inter_AB, inter_AB), dim=1)
        x, m = self.decoder(inter_concat)

        return x, m


class DFUDT(nn.Module):
    def __init__(self, resolution: int = 256, e_dims: int = 64, ae_dims: int = 256, d_dims: int = 64, d_mask_dims: int = 24) -> None:
        super().__init__()

        self.resolution = resolution
        self.e_dims = e_dims
        self.ae_dims = ae_dims
        self.d_dims = d_dims
        self.d_mask_dims = d_mask_dims

        self.encoder = Encoder(in_ch=3, e_ch=e_dims)
        encoder_out_ch = self.encoder.get_output_length(input_resolution=resolution)

        lowest_dense_res = resolution // 32
        self.inter = Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims * 2, lowest_dense_res=lowest_dense_res)
        inter_out_ch = self.inter.get_out_ch()

        self.decoder_src = Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims)
        self.decoder_dst = Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        code = self.encoder(x)
        inter = self.inter(code)
        x, _ = self.decoder_src(inter)
        _, m = self.decoder_dst(inter)

        return x, m


ARCH_SCHEMAS: dict[str, dict] = {
    "LIAE": {
        "model_cls": LIAEUDT,
        "modules": ("encoder", "inter_B", "inter_AB", "decoder"),
    },
    "DF": {
        "model_cls": DFUDT,
        "modules": ("encoder", "inter", "decoder_src", "decoder_dst"),
    },
}


def export_onnx(ckpt: str, sp: str, name: str = "dfm", device: torch.device = torch.device("cuda", 0)) -> None:

    from pathlib import Path
    import onnx
    import io

    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint file: {ckpt} Not found")

    Path(sp).mkdir(exist_ok=True, parents=True)

    print(f" Loading weights from: {ckpt}")
    ckpt = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)

    required_keys = ("iter", "resolution", "e_dims", "ae_dims", "d_dims", "d_mask_dims")
    missing = set(required_keys) - ckpt.keys()
    if missing:
        raise KeyError(f"weights missing keys: {sorted(missing)}")

    values = {k: ckpt[k] for k in required_keys}

    print(" | ".join(f"{k}={values[k]:<6}" for k in required_keys))

    _, resolution, e_dims, ae_dims, d_dims, d_mask_dims = (values[k] for k in required_keys)
    model = LIAEUDT(resolution=resolution, e_dims=e_dims, ae_dims=ae_dims, d_dims=d_dims, d_mask_dims=d_mask_dims)

    for module_name in ["encoder", "inter_B", "inter_AB", "decoder"]:
        if module_name in ckpt:
            print(f"load '{module_name}' from Checkpoint")
            getattr(model, module_name).load_state_dict(ckpt[module_name])
        else:
            warnings.warn(f"missing '{module_name}' weights, skipping load")

    class NHWCWrapper(nn.Module):
        def __init__(self, model: LIAEUDT) -> None:
            super().__init__()
            self.model = model

        def forward(self, nhwc: Tensor) -> Tuple[Tensor, Tensor]:
            nchw = nhwc.permute(0, 3, 1, 2)
            x, m = self.model(nchw)
            x = x.permute(0, 2, 3, 1)
            m = m.permute(0, 2, 3, 1)
            return x, m

    nhwc_model = NHWCWrapper(model)

    nhwc_model.to(device)
    nhwc_model.eval()

    x = torch.rand((1, resolution, resolution, 3), device=device)
    f = io.BytesIO()

    torch.onnx.export(
        nhwc_model,
        x,
        f,
        opset_version=18,
        export_params=True,
        dynamo=True,
        fallback=False,
        optimize=True,
        verify=True,
        external_data=False,
        keep_initializers_as_inputs=False,
        verbose=True,
        # report=True,
        profile=True,
    )

    onnx_model = onnx.load_model_from_string(f.getvalue())

    onnx.checker.check_model(onnx_model, full_check=True)

    # Metadata
    d = {
        "resolution": resolution,
        "e_dims": e_dims,
        "ae_dims": ae_dims,
        "d_dims": d_dims,
        "d_mask_dims": d_mask_dims,
    }
    for k, v in d.items():
        meta = onnx_model.metadata_props.add()
        meta.key, meta.value = k, str(v)

    sp = Path(sp) / f"{name}_{resolution}x{resolution}.onnx"
    onnx.save(onnx_model, sp)
    print(f"✅ 模型成功导出到: {sp}")


if __name__ == "__main__":
    # model = LIAEUDT(resolution=256, e_dims=64, ae_dims=256, d_dims=64, d_mask_dims=24)

    # x = torch.randn(2, 3, 256, 256)

    # enc = model.encoder(x)
    # inter_B = model.inter_B(enc)
    # inter_AB = model.inter_AB(enc)
    # inter_concat = torch.cat([inter_AB, inter_B], dim=1)
    # out, mask = model.decoder(inter_concat)

    # print(f"Output image: {out.shape}")
    # print(f"Output mask: {mask.shape}")

    export_onnx(ckpt="log/dfm_test/ckpt/30000.pth", sp="")
