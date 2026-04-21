from typing import Literal
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.vgg import vgg19, VGG19_Weights
from kornia.color import rgb_to_lab
from dfm.model import EPS

VGG19_Layer_Name = [
    "conv1_1",
    "relu1_1",
    "conv1_2",
    "relu1_2",
    "pool1",
    "conv2_1",
    "relu2_1",
    "conv2_2",
    "relu2_2",
    "pool2",
    "conv3_1",
    "relu3_1",
    "conv3_2",
    "relu3_2",
    "conv3_3",
    "relu3_3",
    "conv3_4",
    "relu3_4",
    "pool3",
    "conv4_1",
    "relu4_1",
    "conv4_2",
    "relu4_2",
    "conv4_3",
    "relu4_3",
    "conv4_4",
    "relu4_4",
    "pool4",
    "conv5_1",
    "relu5_1",
    "conv5_2",
    "relu5_2",
    "conv5_3",
    "relu5_3",
    "conv5_4",
    "relu5_4",
    "pool5",
]


class VGG19(nn.Module):
    def __init__(self, layer_names: list[str], use_input_norm: bool = True):
        super().__init__()

        vgg_features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.selected_layers = [VGG19_Layer_Name.index(l) for l in layer_names]
        self.selected_layers.sort()

        max_idx = self.selected_layers[-1] + 1
        self.features = vgg_features[: max_idx + 1]

    def forward(self, x: Tensor) -> list[Tensor]:
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        features: list[Tensor] = []

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.selected_layers:
                features.append(x)

        return features


class PerceptualLoss(nn.Module):
    """
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
                Here is an example: {'conv5_4': 1.}, which means the conv5_4
                feature layer (before relu5_4) will be extracted with weight
                1.0 in calculting losses.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
    """

    def __init__(self, layer_weights: dict[str, float], criterion: Literal["l1", "mse"] = "l1"):
        super().__init__()

        self.vgg = VGG19(list(layer_weights.keys()))
        self.vgg.requires_grad_(False)
        self.vgg.eval()

        self.vgg = torch.compile(self.vgg, fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True})

        match criterion:
            case "l1":
                self.criterion = F.l1_loss
            case "mse":
                self.criterion = F.mse_loss
            case _:
                raise NotImplementedError(f"{criterion} criterion has not been supported. Only 'l1' and 'mse' are supported.")

        self.register_buffer("weights", torch.tensor(list(layer_weights.values())))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        fx: list[Tensor] = self.vgg(x)
        fy: list[Tensor] = self.vgg(y)

        loss = 0.0
        for weight, a, b in zip(self.weights, fx, fy):
            loss += weight * self.criterion(a, b, reduction="mean")

        return loss


class DSSIMLoss(nn.Module):
    def __init__(self, weight: float = 1.0, window_size: int = 11, sigma: float = 1.5, reduction: Literal["none", "mean", "sum"] = "none"):
        super().__init__()

        assert window_size % 2 == 1, "window_size must be odd"

        self.C = 3

        self.weight = weight
        self.window_size = window_size
        self.sigma = sigma
        self.reduction = reduction

        self.register_buffer("window", self._create_window(window_size, sigma).expand(self.C, 1, self.window_size, self.window_size))

    @staticmethod
    def _gaussian_1d(window_size: int, sigma: float) -> Tensor:
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g

    def _create_window(self, window_size: int, sigma: float) -> Tensor:
        g1d = self._gaussian_1d(window_size, sigma)
        g2d = torch.outer(g1d, g1d)
        window = g2d[None, None, :, :]  # (1,1,ks,ks)
        return window

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:
        """
        返回 SSIM map: (N, 1, H, W)
        """

        C = self.C
        window = self.window

        # 使用 group conv，逐通道独立计算
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=C)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=self.window_size // 2, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=self.window_size // 2, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=C) - mu_xy

        sigma_x2 = torch.clamp(sigma_x2, min=0.0)
        sigma_y2 = torch.clamp(sigma_y2, min=0.0)

        # SSIM 常量（假设输入范围 [0,1]）
        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map: Tensor = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + EPS)

        # RGB → mean over channel
        return ssim_map.mean(dim=1, keepdim=True)

    @torch.compile(mode="max-autotune")
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x, y: (N,3,H,W), float in [0,1]
        """
        # assert x.shape == y.shape, "Input shapes must match"
        # assert x.dim() == 4 and x.size(1) == 3, "Expect NCHW RGB input"

        ssim = self._ssim(x, y)  # (N,1,H,W)
        dssim = (1.0 - ssim) * 0.5 * self.weight  # (N,1,H,W)

        match self.reduction:
            case "none":
                return dssim
            case "mean":
                return dssim.mean()
            case "sum":
                return dssim.sum()
            case _:
                raise ValueError(f"Invalid reduction: {self.reduction}")


def mean_std(x: Tensor) -> tuple[Tensor, Tensor]:
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    std = torch.sqrt(var + EPS)
    return mean, std


def style_loss_mean_std(pred: Tensor, target: Tensor, weight: float = 1.0) -> Tensor:

    m_p, s_p = mean_std(pred)
    m_s, s_s = mean_std(target)

    loss = F.l1_loss(m_p, m_s) + F.l1_loss(s_p, s_s)
    return loss * weight


class StyleLossLabChroma(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()

        self.weight = weight

    @torch.compile(mode="max-autotune")
    def forward(self, pred_rgb: Tensor, target_rgb: Tensor) -> Tensor:
        pred_lab = rgb_to_lab(pred_rgb)
        target_lab = rgb_to_lab(target_rgb)

        pred_ab = pred_lab[:, 1:3]
        target_ab = target_lab[:, 1:3]

        return style_loss_mean_std(pred_ab, target_ab, self.weight)


class PixelLoss(nn.Module):
    def __init__(self, weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "none"):
        super().__init__()

        self.reduction = reduction
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:

        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return loss * self.weight


class MaskLoss(nn.Module):
    def __init__(self, weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()

        self.reduction = reduction
        self.weight = weight

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = F.binary_cross_entropy(pred, target, reduction=self.reduction)
        return loss * self.weight


class DLoss(nn.Module):
    def __init__(self, weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "none"):
        super().__init__()

        self.reduction = reduction
        self.weight = weight

    def forward(self, fake_score: Tensor, real_score: Tensor) -> Tensor:

        loss_real = torch.relu(1.0 - real_score)
        loss_fake = torch.relu(1.0 + fake_score)

        loss = self.weight * (loss_real + loss_fake) * 0.5

        match self.reduction:
            case "none":
                return loss
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()


class GANLoss(nn.Module):
    def __init__(self, weight: float = 1.0, reduction: Literal["none", "mean", "sum"] = "none"):
        super().__init__()

        self.reduction = reduction
        self.weight = weight

    def forward(self, pred_score: Tensor) -> Tensor:
        loss = self.weight * -pred_score

        match self.reduction:
            case "none":
                return loss
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
