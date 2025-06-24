import os
import time
import itertools
from torchvision import utils
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from losses.arcface import ArcFace
from losses.losses import MultiScaleGANLoss
from model import FullGenerator, StyleGAN2Discriminator, MultiScaleDiscriminator
from losses import GANLoss, L1Loss, PerceptualLoss, r1_penalty, g_path_regularize
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader.data_loader import FaceDataset

assert torch.cuda.is_available()


class TrainSet:

    device: torch.device = torch.device("cuda", 0)
    pretrain: None | str = "ckpt/1129103_512.pth"

    batch: int = 12
    size: int = 512
    style_dim: int = 512
    n_mlp: int = 8
    isconcat: bool = True

    save_samples_every: int = 250
    save_checkpoint_every: int = 5000
    grad_accum_steps: int = 1

    r1_reg_every: int = 16
    r1_reg_weight: float = 10.0

    path_reg_every: int = 4
    path_reg_weight: float = 2.0

    d_lr: float = 0.0001
    g_lr: float = 0.0001

    d_optimiz_every: int = 1
    g_optimiz_every: int = 1

    ckpt_path: str = "ckpt"
    sample_path: str = "sample"
    log_path: str = "log/train"

    id_loss_weight: float = 4.0

    gan_type: str = "wgan_softplus"
    gan_loss_weight: float = 1.0

    l1_loss_weight: float = 10.0
    l1_reduction: str = "mean"

    PerceptualLoss_loss_type: str = "vgg19"
    PerceptualLoss_layer_weights: dict[str, float] = {
        "conv3_4": 1.0,
        # "relu3_4": 1.0,
        # "relu4_3": 1.0
        "conv4_4": 1.0,
    }
    PerceptualLoss_use_input_norm: bool = True
    PerceptualLoss_range_norm: bool = True
    PerceptualLoss_perceptual_weight: float = 10.0
    PerceptualLoss_style_weight: float = 10.0
    PerceptualLoss_criterion: str = "l1"

    mask_loss_weight: float = 1.0

    same_image_prob: float = 0.15

    # "/home/liaohaixun/styleswap_data/dateset/ffhq_1024",
    # "/home/liaohaixun/styleswap_data/dateset/CelebAHQ-1024x1024",
    # "/home/liaohaixun/styleswap_data/dateset/vggface2_hq512",
    # "filter_results/closed_eyes"
    # "filter_results/large_angle"

    image_path = None

    target_dirs = [
        ("/home/liaohaixun/styleswap_data/dateset/ffhq_1024", 0.25),
        ("/home/liaohaixun/styleswap_data/dateset/CelebAHQ-1024x1024", 0.25),
        ("filter_results/closed_eyes", 0.25),
        ("filter_results/large_angle", 0.25),
    ]
    source_dirs = [
        ("/home/liaohaixun/styleswap_data/dateset/ffhq_1024", 0.5),
        ("/home/liaohaixun/styleswap_data/dateset/CelebAHQ-1024x1024", 0.5),
    ]


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, config: TrainSet):

        config.r1_reg_every = config.r1_reg_every * config.grad_accum_steps
        config.path_reg_every = config.path_reg_every * config.grad_accum_steps

        config.r1_reg_every = config.r1_reg_every * config.d_optimiz_every
        config.path_reg_every = config.path_reg_every * config.g_optimiz_every

        self.cfg = config
        self.iter = 1

        torch.cuda.set_device(self.cfg.device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        self.loss_dict = {}

        self.net_d = MultiScaleDiscriminator(out_size=self.cfg.size).to(self.cfg.device)

        self.net_g = FullGenerator(
            self.cfg.size,
            style_dim=self.cfg.style_dim,
            n_mlp=self.cfg.n_mlp,
            isconcat=self.cfg.isconcat,
        ).to(self.cfg.device)
        self.net_g_ema = FullGenerator(
            self.cfg.size,
            style_dim=self.cfg.style_dim,
            n_mlp=self.cfg.n_mlp,
            isconcat=self.cfg.isconcat,
        ).to(self.cfg.device)

        if self.cfg.pretrain is not None:
            print(f"Loading pretrain checkpoint from {self.cfg.pretrain}")

            checkpoint = torch.load(
                self.cfg.pretrain,
                map_location=self.cfg.device,
                weights_only=False,
            )

            self.iter = checkpoint.get("iter", 0)
            print(f"iter = {self.iter}")

            self.net_d.load_state_dict(checkpoint["d"])
            self.net_g.load_state_dict(checkpoint["g"])
            self.net_g_ema.load_state_dict(checkpoint["g_ema"])
            print("Pretrain checkpoint loaded successfully.")

        self.net_d.train()
        self.net_g.train()
        self.net_g_ema.eval()

        self.d_optimizer = torch.optim.Adam(
            params=self.net_d.parameters(),
            lr=self.cfg.d_lr,
            betas=(0.0, 0.99),
        )

        self.g_optimizer = torch.optim.Adam(
            params=self.net_g.parameters(),
            lr=self.cfg.g_lr,
            betas=(0.0, 0.99),
        )

        if self.cfg.pretrain is not None:
            self.d_optimizer.load_state_dict(checkpoint["d_optim"])
            for param_group in self.d_optimizer.param_groups:
                param_group["lr"] = self.cfg.d_lr

            self.g_optimizer.load_state_dict(checkpoint["g_optim"])
            for param_group in self.g_optimizer.param_groups:
                param_group["lr"] = self.cfg.g_lr
            print("Optimizer state loaded from checkpoint.")

        print(f"d_optimizer_lr: {self.d_optimizer.param_groups[0]["lr"]}")
        print(f"g_optimizer_lr: {self.g_optimizer.param_groups[0]["lr"]}")

        Path(self.cfg.ckpt_path).mkdir(exist_ok=True, parents=True)
        Path(self.cfg.sample_path).mkdir(exist_ok=True, parents=True)
        Path(self.cfg.log_path).mkdir(exist_ok=True, parents=True)

        self.gan_loss = MultiScaleGANLoss(
            self.cfg.gan_type, loss_weight=self.cfg.gan_loss_weight
        ).to(self.cfg.device)
        self.l1_loss = L1Loss(self.cfg.l1_loss_weight, self.cfg.l1_reduction).to(
            self.cfg.device
        )
        self.perceptualLoss = PerceptualLoss(
            self.cfg.PerceptualLoss_layer_weights,
            self.cfg.PerceptualLoss_loss_type,
            self.cfg.PerceptualLoss_use_input_norm,
            self.cfg.PerceptualLoss_range_norm,
            self.cfg.PerceptualLoss_perceptual_weight,
            self.cfg.PerceptualLoss_style_weight,
            self.cfg.PerceptualLoss_criterion,
        ).to(self.cfg.device)

        self.arcface = ArcFace().to(self.cfg.device)
        self.arcface.eval()
        self.arcface.requires_grad_(False)

        dataset = FaceDataset(
            size=self.cfg.size,
            target_dirs=self.cfg.target_dirs,
            source_dirs=self.cfg.source_dirs,
            same_image_prob=self.cfg.same_image_prob,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True,
            pin_memory_device=str(self.cfg.device),
            prefetch_factor=3,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        self.loader = sample_data(loader)

        self.writer = SummaryWriter(log_dir=self.cfg.log_path)

    def model_ema(self, decay=0.999):

        net_g_params = dict(self.net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(
                net_g_params[k].data, alpha=1 - decay
            )

    def save(self):
        save_dict = {
            "iter": self.iter,
            "d": self.net_d.state_dict(),
            "g": self.net_g.state_dict(),
            "g_ema": self.net_g_ema.state_dict(),
            "d_optim": self.d_optimizer.state_dict(),
            "g_optim": self.g_optimizer.state_dict(),
        }

        save_file = os.path.join(
            self.cfg.ckpt_path, f"{str(self.iter)}_{self.cfg.size}.pth"
        )

        torch.save(save_dict, save_file)

    def train(self):

        cfg = self.cfg

        pbar = tqdm(itertools.count(start=self.iter), initial=self.iter, unit="iter")

        loss_log = os.path.join(cfg.log_path, "loss.txt")

        # mean_path_length = 0

        # 主训练循环
        for self.iter in pbar:

            # 是否为累积梯度步数
            is_accum_step = self.iter % cfg.grad_accum_steps == 0

            # 获取当前批次数据
            target, source, mask, is_same = next(self.loader)
            target = target.to(cfg.device)
            source = source.to(cfg.device)
            mask = mask.to(cfg.device)
            is_same = is_same.to(cfg.device)

            # 提取身份特征
            with torch.no_grad():
                z_id = self.arcface(source)

            # =============== 训练判别器 ===============
            if self.iter % cfg.d_optimiz_every == 0:

                self.net_d.requires_grad_(True)
                self.net_g.requires_grad_(False)

                # 生成假图像
                fake_img, _, _ = self.net_g(target, z_id.detach())

                # 判别真实图像
                fake_pred, _ = self.net_d(fake_img.detach())
                real_pred, _ = self.net_d(target)

                # 计算判别器损失
                l_d = (
                    self.gan_loss(real_pred, True, is_disc=True)
                    + self.gan_loss(fake_pred, False, is_disc=True)
                ) / cfg.grad_accum_steps

                # 记录损失
                self.loss_dict["l_d"] = l_d.detach()
                # self.loss_dict["real_score"] = real_pred.detach().mean()
                for i, pred in enumerate(real_pred):
                    self.loss_dict[f"real_score_s{i}"] = pred.detach().mean()

                for i, pred in enumerate(fake_pred):
                    self.loss_dict[f"fake_score_s{i}"] = pred.detach().mean()
                # self.loss_dict["fake_score"] = fake_pred.detach().mean()

                # R1正则化
                if self.iter % cfg.r1_reg_every == 0:
                    target.requires_grad_(True)

                    real_pred, downsampled = self.net_d(target, return_downsampled=True)
                    l_d_r1 = 0.0
                    for r, t in zip(real_pred, downsampled):
                        l_d_r1_sig = r1_penalty(r, t)
                        l_d_r1_sig = (
                            cfg.r1_reg_weight / 2 * l_d_r1_sig * cfg.r1_reg_every
                        ) / cfg.grad_accum_steps
                        l_d_r1 += l_d_r1_sig

                    l_d_r1 = l_d_r1 / len(downsampled)
                    self.loss_dict["l_d_r1"] = l_d_r1.detach().mean()
                    l_d += l_d_r1

                # 反向传播
                l_d.backward()
                if is_accum_step:
                    self.d_optimizer.step()
                    self.d_optimizer.zero_grad()

            # =============== 训练生成器 ===============
            if self.iter % cfg.g_optimiz_every == 0:

                self.net_d.requires_grad_(False)
                self.net_g.requires_grad_(True)

                # 生成假图像
                fake_img, _, fake_mask = self.net_g(target, z_id.detach())
                fake_pred, _ = self.net_d(fake_img)

                # ------------- 对抗损失 -------------
                l_g = self.gan_loss(fake_pred, True, is_disc=False)
                self.loss_dict["l_g_wgan"] = l_g.detach()

                # ------------- 身份损失 -------------
                fake_z_id = self.arcface(fake_img)
                id_loss = (1 - torch.cosine_similarity(fake_z_id, z_id.detach())).mean()
                self.loss_dict["id"] = id_loss.detach()
                l_g += id_loss * cfg.id_loss_weight

                # ------------- 感知损失 -------------
                l_g_percep, l_g_style = self.perceptualLoss(fake_img, target)
                if l_g_percep is not None:
                    l_g += l_g_percep
                    self.loss_dict["l_g_percep"] = l_g_percep.detach()
                if l_g_style is not None:
                    l_g += l_g_style
                    self.loss_dict["l_g_style"] = l_g_style.detach()

                # ------------- 启发式误差识别细化(FaceShifter) -------------
                # l_g_pix = 0.0
                # for i, flag in enumerate(is_same):
                #     if flag:
                #         l_i = self.l1_loss(fake_img[i], target[i])
                #     else:
                #         diff_i = self.l1_loss(fake_img[i] * non_face_mask[i],
                #                             target[i] * non_face_mask[i])
                #         # 注意：mean 会默认除以全图像素数，你可能想自己做 sum/count
                #         l_i = diff_i * (C*H*W) / non_face_mask[i].sum().clamp_min(1.0)
                #     l_g_pix += l_i

                B, C, H, W = fake_img.shape
                # 1. 全图绝对误差 & 非脸区域误差
                full_diff = torch.abs(fake_img - target)  # [B,C,H,W]
                non_face_mask = (1 - mask).expand_as(full_diff)  # [B,C,H,W]
                non_face_diff = full_diff * non_face_mask  # [B,C,H,W]

                # 2. 求和与计数
                sum_full = full_diff.reshape(B, -1).sum(dim=1)  # [B]
                sum_non = non_face_diff.reshape(B, -1).sum(dim=1)  # [B]
                count_non = non_face_mask.reshape(B, -1).sum(dim=1)  # [B]
                count_non = count_non.clamp_min(1.0)  # 避免 0

                # 3. 分样本 L1
                num_pixels = float(C * H * W)
                loss_per_sample = torch.where(
                    is_same, sum_full / num_pixels, sum_non / count_non
                )  # [B]

                # 4. batch 平均
                l_g_pix = loss_per_sample.mean()
                l_g_pix *= cfg.l1_loss_weight

                l_g += l_g_pix
                self.loss_dict["l_g_pix"] = l_g_pix

                # ------------- 掩码损失 -------------
                mask_loss = F.binary_cross_entropy(fake_mask, mask).mean()
                self.loss_dict["l_g_mask"] = mask_loss.detach()
                l_g += mask_loss * cfg.mask_loss_weight

                # ------------- 总损失 -------------
                g_loss = l_g / cfg.grad_accum_steps

                # 路径正则化
                # if self.iter % cfg.path_reg_every == 0:

                #     fake_img, latents, fake_mask = self.net_g(
                #         target, z_id, return_latents=True
                #     )

                #     path_loss, mean_path_length, path_lengths = g_path_regularize(
                #         fake_img, latents, mean_path_length
                #     )

                #     weighted_path_loss = (
                #         cfg.path_reg_weight * cfg.path_reg_every * path_loss
                #     ) / cfg.grad_accum_steps
                #     # weighted_path_loss += 0.0 * fake_img[0, 0, 0, 0]
                #     # weighted_path_loss += 0.0 * fake_mask[0, 0, 0, 0]
                #     g_loss += weighted_path_loss

                #     self.loss_dict["path_loss"] = path_loss
                #     self.loss_dict["path_lengths"] = path_lengths.mean()

                # 反向传播
                g_loss.backward()
                if is_accum_step:
                    self.g_optimizer.step()
                    self.g_optimizer.zero_grad()
                    self.model_ema()

            # 统计所有损失和分数
            values = {}
            for k, v in self.loss_dict.items():
                # v 可能是张量或元组
                if isinstance(v, tuple):
                    v = v[0]
                if torch.is_tensor(v):
                    values[k] = v.mean().item()
                else:
                    values[k] = float(v)

            # 更新进度条描述
            # desc = "; ".join([f"{k}: {values[k]:.4f}" for k in values])
            # pbar.set_description(desc)

            # 记录到Tensorboard
            for k, v in values.items():
                self.writer.add_scalar(f"Loss/{k}", v, self.iter)

            # 定期保存样本和日志
            if self.iter % cfg.save_samples_every == 0:

                # 保存损失日志
                desc = "; ".join([f"{k}: {values[k]:.4f}" for k in values])
                with open(loss_log, "a") as f:
                    f.write(f"itr: {self.iter}; {desc}\n")

                # 生成样本图像
                with torch.no_grad():
                    fake, _, fake_mask = self.net_g_ema(target, z_id)

                    fake = fake * fake_mask + (1 - fake_mask) * target

                    fake_mask = (fake_mask.repeat(1, 3, 1, 1) - 0.5) / 0.5

                    mask = (mask.repeat(1, 3, 1, 1) - 0.5) / 0.5

                    grid = torch.cat((target, source, fake, fake_mask, mask), 0)

                    save_file = os.path.join(
                        cfg.sample_path, f"{str(self.iter)}_{cfg.size}.png"
                    )

                    # 保存样本图像
                    utils.save_image(
                        grid,
                        save_file,
                        nrow=cfg.batch,
                        normalize=True,
                        value_range=(-1, 1),
                    )

            # 定期保存模型
            if self.iter % cfg.save_checkpoint_every == 0:
                self.save()

        print("训练完成!")


if __name__ == "__main__":

    cfg = TrainSet()
    trainer = Trainer(cfg)
    try:
        trainer.train()
    finally:
        trainer.save()
        trainer.writer.close()
        torch.cuda.empty_cache()
