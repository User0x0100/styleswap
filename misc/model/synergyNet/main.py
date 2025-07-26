from pathlib import Path
import types
import torch
from .backbone_nets.pointnet_backbone import MLP_for, MLP_rev
from .backbone_nets.mobilenetv2_backbone import MobileNetV2
from torch import nn

from .utils.params import ParamsPack

param_pack = ParamsPack()


class SynergyNet_impl(nn.Module):
    """
    # https://github.com/choyingw/SynergyNet
    """

    def __init__(self):

        super().__init__()

        self.backbone = MobileNetV2()
        self.forwardDirection = MLP_for(68)
        self.reverseDirection = MLP_rev(68)

        self.register_buffer("param_mean", torch.Tensor(param_pack.param_mean))
        self.register_buffer("param_std", torch.Tensor(param_pack.param_std))
        self.register_buffer("w_shp", torch.Tensor(param_pack.w_shp))
        self.register_buffer("u", torch.Tensor(param_pack.u))
        self.register_buffer("w_exp", torch.Tensor(param_pack.w_exp))

        # Online training needs these to parallel
        self.register_buffer("u_base", torch.Tensor(param_pack.u_base))
        self.register_buffer("w_shp_base", torch.Tensor(param_pack.w_shp_base))
        self.register_buffer("w_exp_base", torch.Tensor(param_pack.w_exp_base))
        self.keypoints = torch.Tensor(param_pack.keypoints).long()

        state_dict = torch.load(Path(__file__).with_name("model.pt"), map_location=torch.device("cpu"), weights_only=False)["state_dict"]

        state_dict = {
            cleaned: v for k, v in state_dict.items() if (cleaned := k.removeprefix("module.").removeprefix("I2P.")) not in ("u_tex", "w_tex")
        }  # 去掉 "module. I2P. 排除顶层 u_tex w_tex"

        self.load_state_dict(state_dict, strict=True)

    def forward(self, input):
        _3D_attr, avgpool = self.backbone(input)
        return _3D_attr, avgpool

    @staticmethod
    def parse_param_62(param):
        """Work for only tensor"""
        p_ = param[:, :12].reshape(-1, 3, 4)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].reshape(-1, 3, 1)
        alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
        alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
        return p, offset, alpha_shp, alpha_exp

    def reconstruct_vertex_62(self, param, whitening=True, dense=False, transform=True, lmk_pts=68):
        """
        Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
        dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
        image coordinate space, but without alignment caused by face cropping.
        transform: whether transform to image space
        Working with batched tensors. Using Fortan-type reshape.
        """

        if whitening:
            if param.shape[1] == 62:
                param_ = param * self.param_std[:62] + self.param_mean[:62]
            else:
                raise RuntimeError("length of params mismatch")

        p, offset, alpha_shp, alpha_exp = self.parse_param_62(param_)

        if dense:

            vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).contiguous().view(-1, 53215, 3).transpose(1, 2) + offset

            if transform:
                # transform to image coordinate space
                vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        else:
            """For 68 pts"""
            vertex = p @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp).contiguous().view(-1, lmk_pts, 3).transpose(1, 2) + offset

            if transform:
                # transform to image coordinate space
                vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        return vertex


def draw_landmarks_on_tensor(images, landmarks, color=(0, 1, 0), radius=1):
    """
    使用PyTorch在图像tensor上绘制关键点
    Args:
        images: 图像tensor [B, C, H, W]，值范围[-1, 1]
        landmarks: 关键点坐标tensor [B, 3, 68]
        color: 关键点颜色 (RGB, 0-1范围)
        radius: 关键点半径
    Returns:
        绘制了关键点的图像tensor
    """
    B, C, H, W = images.shape
    device = images.device

    # 转换图像到[0, 1]范围用于显示
    images_viz = (images + 1) / 2
    # images_viz = images

    # 创建结果tensor
    result = images_viz.clone()

    # 为每个batch绘制关键点
    for b in range(B):
        lmks = landmarks[b]  # [3, 68]

        # 使用x, y坐标（忽略z坐标）
        x_coords = lmks[0].clamp(0, W - 1).long()  # [68]
        y_coords = lmks[1].clamp(0, H - 1).long()  # [68]

        # 为每个关键点绘制圆形
        for i in range(68):
            x, y = x_coords[i].item(), y_coords[i].item()

            # 创建圆形mask
            y_grid, x_grid = torch.meshgrid(
                torch.arange(max(0, y - radius), min(H, y + radius + 1), device=device),
                torch.arange(max(0, x - radius), min(W, x + radius + 1), device=device),
                indexing="ij",
            )

            # 计算距离并创建圆形mask
            dist = ((x_grid - x) ** 2 + (y_grid - y) ** 2).float()
            circle_mask = dist <= radius**2

            if circle_mask.any():
                # 应用颜色到所有通道
                for c in range(C):
                    result[b, c, y_grid[circle_mask], x_grid[circle_mask]] = color[c]

    return result


if __name__ == "__main__":

    import random
    import torchvision.utils as utils
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import functional as F

    class TranslateUp:
        def __init__(self, pixels: int):
            # pixels: 向上移动的像素数量
            self.pixels = pixels

        def __call__(self, img):
            # 参数 translate=(tx, ty)，其中 ty 为正时向下移动，ty<0 向上移动
            return F.affine(img, angle=0, translate=(0, -self.pixels), scale=1.0, shear=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    transform = transforms.Compose(
        [
            transforms.Resize(130),
            # TranslateUp(pixels=13),
            transforms.CenterCrop(120),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    print(f"使用设备: {device}")

    images = [Image.open(f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png").convert("RGB") for _ in range(batch_size)]
    # images_raw = [transforms.ToTensor()(img) for img in images]
    # images_raw = torch.stack(images_raw).to(device=device)
    # scale = images[0].width / 120
    images = [transform(img) for img in images]
    images = torch.stack(images).to(device=device)

    # swap R B channel
    images = images[:, [2, 1, 0], :, :]

    net = SynergyNet_impl().to(device)
    net.eval()

    with torch.no_grad():
        _3D_attr, _ = net(images)
        lmks = net.reconstruct_vertex_62(_3D_attr)
        print(f"关键点形状: {lmks.shape}")

        # lmks *= scale

    # 使用PyTorch绘制关键点
    images_with_landmarks = draw_landmarks_on_tensor(images[:, [2, 1, 0], :, :], lmks, color=(0, 1, 0), radius=1)

    # 使用torchvision创建网格并保存
    grid = utils.make_grid(images_with_landmarks, nrow=4, padding=2, normalize=False)
    utils.save_image(grid, "landmarks_grid_pytorch.png")

    print(f"处理了 {batch_size} 张图片")
    print(f"每张图片检测到 {lmks.shape[2]} 个关键点")
    print("关键点可视化结果已保存到 landmarks_grid_pytorch.png")
