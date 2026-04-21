import itertools
from pathlib import Path
from typing import Any

import cv2
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from torch import Tensor
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid

from losses import IDLoss, L1Loss
from dfm.losses import PerceptualLoss, DLoss, GANLoss
from misc.model import IDEncoder


from .dataloader import datasetloader
from .networks.generator import Generator
from .networks.discriminator import Discriminator


EPS = 1e-6

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def ifsr_loss_func(bs: int, src_ifsr_feats: dict[str, Tensor], dst_ifsr_feats: dict[str, Tensor], ifsr_dict: dict[str, float]):
    loss = 0.0
    for layer_name, margin in ifsr_dict.items():

        feat_true_flat = src_ifsr_feats[layer_name].reshape(bs, -1)
        feat_pred_flat = dst_ifsr_feats[layer_name].reshape(bs, -1)
        distance = (1.0 - F.cosine_similarity(feat_true_flat, feat_pred_flat, dim=1)).mean()

        d_loss = F.relu(distance - margin)
        loss += d_loss

    return loss


class Trainer:
    def __init__(
        self,
        src: list[tuple[str, float]],
        dst: list[tuple[str, float]],
        batch_size: int = 10,
        lr: float = 1e-4,
        lr_scheduler_t_max: int = 0,
        bf16: bool = True,
        device: str = "cuda:0",
        compile_module: bool = True,
        weight: str | None = None,
        log_path: str = "log_dfm/train_facedancer",
        sample_save_every: int = 1000,
        weight_save_every: int = 10000,
        # 模型配置
        size: int = 256,
        # 损失配置
        ifsr_scale: float = 1.2,
        ifsr_dict: dict[str, float] = {
            "layer3.5": 0.121357,
            "layer3.4": 0.128827,
            "layer3.3": 0.117972,
            "layer3.2": 0.109391,
            "layer3.1": 0.097296,
            "layer3.0": 0.089046,
            "layer2.3": 0.044928,
            "layer2.2": 0.048719,
            "layer2.1": 0.047487,
            "layer2.0": 0.047970,
            "layer1.2": 0.035144,
        },  # "layer_name": margin
        id_encode_provider: IDEncoder.PROVIDER = IDEncoder.PROVIDER.BLENDFACE,
        id_loss_weight: float = 10.0,
        rec_loss: float = 5.0,
        vgg19_loss_weight: dict[str, float] = {
            "pool1": 0.2,
            "pool2": 0.2,
            "pool3": 0.2,
            "pool4": 0.2,
            "pool5": 0.2,
        },
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.bf16 = bf16
        self.sample_save_every = sample_save_every
        self.weight_save_every = weight_save_every
        self.enable_lr_scheduler = lr_scheduler_t_max > 0
        self.ifsr_dict = ifsr_dict

        # ========================= Init Model =========================

        if weight is not None:
            if Path(weight).exists() == False:
                raise FileNotFoundError(f"Weight file: {weight} Not found")

            print(f"Loading weight from {weight}")
            weight: dict[str, Any] = torch.load(weight, map_location=torch.device("cpu"), weights_only=False)

            self.iter, self.size = (weight[k] for k in ("iter", "size"))

            print(f"Weight Info:\n" f"  {'iter':25}: {self.iter}\n" f"  {'size':25}: {self.size}\n")

            self.net_g = Generator(input_res=self.size)
            self.net_d = Discriminator(input_res=self.size)
            self.net_g.load_state_dict(weight["net_g"])
            self.net_d.load_state_dict(weight["net_d"])

        else:

            self.iter, self.size = 0, size
            self.net_g = Generator(input_res=self.size)
            self.net_d = Discriminator(input_res=self.size)

        self.net_g.to(self.device)
        self.net_d.to(self.device)
        self.net_g.train()
        self.net_d.train()

        # ========================= Optim =========================
        self.optim_g = optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)
        self.optim_d = optim.Adam(self.net_d.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)

        if self.enable_lr_scheduler:
            self.lr_scheduler_g = CosineAnnealingLR(self.optim_g, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)
            self.lr_scheduler_d = CosineAnnealingLR(self.optim_d, T_max=lr_scheduler_t_max, eta_min=lr * 0.1)

        # ========================= LOSS =========================

        self.d_loss = DLoss(weight=1.0, reduction="mean").to(self.device)
        self.gan_loss = GANLoss(weight=1.0, reduction="mean").to(self.device)

        self.id_loss = IDLoss(weight=id_loss_weight, id_encode_provider=id_encode_provider).to(self.device)
        self.id_loss.register_ifsr_hooks(list(self.ifsr_dict.keys()))
        for k in self.ifsr_dict:
            self.ifsr_dict[k] *= ifsr_scale

        self.rec_loss = L1Loss(loss_weight=rec_loss, reduction="none").to(self.device)
        self.vgg19_loss = PerceptualLoss(layer_weights=vgg19_loss_weight).to(self.device)

        # ========================= LOG =========================
        base_log_path = Path(log_path)
        self.ckpt_dir = base_log_path.joinpath("ckpt")
        self.sample_dir = base_log_path.joinpath("sample")
        self.tb_log_dir = base_log_path.joinpath("tb_log")

        for p in [self.ckpt_dir, self.sample_dir, self.tb_log_dir]:
            p.mkdir(exist_ok=True, parents=True)

        self.log_writer = SummaryWriter(log_dir=self.tb_log_dir)

        # ========================= Sample =========================
        pipe = datasetloader(
            batch_size=self.batch_size,
            num_threads=16,
            prefetch_queue_depth=4,
            py_num_workers=1,
            py_start_method="spawn",
            device_id=self.device.index,
            resize=self.size,
            src=src,
            dst=dst,
            same_image_prob=0.2,
        )

        self.dataset = DALIGenericIterator(pipelines=pipe, output_map=["src", "dst", "is_same"], auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

        # ========================= compile_module =========================

        self.train_module: dict[str, nn.Module] = {}

        for net_name in ["net_g", "net_d"]:
            net: nn.Module = getattr(self, net_name)
            self.train_module[net_name] = torch.compile(net, fullgraph=True, dynamic=False, options={"epilogue_fusion": True, "max_autotune": True}) if compile_module else net

    @torch.no_grad()
    def log(self, k: str, v: Tensor, now: bool = False) -> None:
        if now or self.iter % 10 == 0:
            self.log_writer.add_scalar(f"Loss/{k}", v.detach().mean().item(), self.iter)

    @torch.no_grad()
    def fetch_sample(self) -> tuple[Tensor, Tensor, Tensor]:
        data: dict[str, Tensor] = self.dataset.next()[0]
        src, dst, is_same = (data[k] for k in ("src", "dst", "is_same"))
        is_same = is_same.squeeze(-1)
        return src, dst, is_same

    @torch.no_grad()
    def save_state_dict(self):
        state_dict = {
            "iter": self.iter,
            "size": self.size,
            "net_g": self.net_g.state_dict(),
            "net_d": self.net_d.state_dict(),
        }
        try:
            ckpt_file = self.ckpt_dir / f"{self.iter}.pth"
            torch.save(state_dict, ckpt_file)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def train(self):
        pbar = tqdm(itertools.count(start=self.iter), initial=self.iter, mininterval=1.0, bar_format="{n_fmt:7} | speed {rate_fmt:3} | train time {elapsed}")

        net_d, net_g = self.train_module["net_d"], self.train_module["net_g"]

        for self.iter in pbar:
            src, dst, is_same = self.fetch_sample()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):

                with torch.no_grad():
                    src_id_feats = self.id_loss.get_id_feats(src, None, realign=False).detach_()

                    _ = self.id_loss.get_id_feats(dst, None, realign=False).detach_()
                    dst_ifsr_feats = {k: v.detach_().clone() for k, v in self.id_loss._ifsr_feats.items()}

                torch.compiler.cudagraph_mark_step_begin()
                fake: Tensor = net_g(dst, src_id_feats)

                # ========================= train d =========================
                self.optim_d.zero_grad()

                dst_d = dst.detach().requires_grad_(True)

                fake_score: Tensor = net_d(fake.detach())
                real_score: Tensor = net_d(dst_d)

                d_loss: Tensor = self.d_loss(fake_score, real_score)
                self.log("d_loss", d_loss)

                real_grads = torch.autograd.grad(outputs=real_score.sum(), inputs=dst_d, create_graph=True, retain_graph=True, only_inputs=True)[0]
                gp = real_grads.pow(2).sum(dim=(1, 2, 3))
                gp_loss = (gp * (10.0 * 0.5)).mean()

                d_loss += gp_loss

                d_loss.backward()
                self.optim_d.step()

                # ========================= train g =========================
                self.optim_g.zero_grad()

                loss: float | Tensor
                loss = 0.0

                # loss gan
                fake_score = net_d(fake)
                gan_loss = self.gan_loss(fake_score)
                self.log("gan_loss", gan_loss)
                loss += gan_loss

                # loss id
                fake_id_feats = self.id_loss.get_id_feats(fake, None, realign=False)
                fake_ifsr_feats = {k: v.clone() for k, v in self.id_loss._ifsr_feats.items()}
                id_loss = self.id_loss(fake_id_feats, src_id_feats)
                self.log("id_loss", id_loss)
                loss += id_loss

                # ifsr loss
                ifsr_loss = ifsr_loss_func(self.batch_size, fake_ifsr_feats, dst_ifsr_feats, self.ifsr_dict)
                self.log("ifsr_loss", ifsr_loss)
                loss += ifsr_loss

                # loss vgg19
                vgg19_loss = self.vgg19_loss(fake, dst)
                self.log("vgg19_loss", vgg19_loss)
                loss += vgg19_loss

                # loss rec
                rec_loss: Tensor = self.rec_loss(fake, dst)  # BCHW
                rec_loss = rec_loss.mean(dim=[1, 2, 3])
                rec_loss = (rec_loss * is_same).sum() / is_same.sum().clamp_min(1.0)
                self.log("rec_loss", rec_loss)
                loss += rec_loss

                loss.backward()
                self.optim_g.step()

                if self.enable_lr_scheduler:
                    self.lr_scheduler_g.step()
                    self.lr_scheduler_d.step()

            if self.iter % self.weight_save_every == 0:
                self.save_state_dict()

            if self.iter % self.sample_save_every == 0:
                with torch.no_grad():
                    attn_map: list[Tensor] = self.net_g.get_attention_maps()

                    maps = []
                    for m in attn_map:
                        m = m.mean(dim=1, keepdim=True)
                        m = F.interpolate(m, size=self.size, mode="bilinear", align_corners=False)
                        maps.append(m)
                    attn = torch.mean(torch.stack(maps, dim=0), dim=0).expand(-1, 3, -1, -1)
                    amin = attn.amin(dim=(2, 3), keepdim=True)
                    amax = attn.amax(dim=(2, 3), keepdim=True)
                    attn_norm = 2.0 * (attn - amin) / (amax - amin + 1e-6) - 1.0

                    grid = torch.cat((src, dst, fake, attn_norm), dim=0)
                    grid.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
                    grid = make_grid(grid, nrow=self.batch_size)[[2, 1, 0], :, :]  # RGB -> BGR
                    grid = grid.permute(1, 2, 0)  # CHW -> HWC
                    grid_cpu = grid.to(device="cpu", dtype=torch.uint8).numpy()
                    cv2.imwrite(self.sample_dir / f"{self.iter}.png", grid_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


if __name__ == "__main__":
    src = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
    ]

    dst = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/RealOcc/image/realign_arcface_dst", 2.0),
        ("/opt/share/deepfake/dataset_1/youtube/What_s_considered_tall_in_South_Korea_Street_Interview_align_results", 0.0),
        ("/opt/share/deepfake/dataset_1/youtube/4k_Face_Close_Up_HDR_Video_Vivid_Colors_Ambient_Sound_-_Relaxing_align_results", 0.0),
        ("/opt/share/deepfake/dataset_1/oneman/1_align_results/", 0.0),
    ]
    trainer = Trainer(src, dst, log_path="log_dfm/train_facedancer_BLENDFACE_512", size=512)

    try:
        trainer.train()
    finally:
        trainer.save_state_dict()
