import itertools
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import torch
import torch.distributed as dist
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader import datasetloader
from face_swap_model import FullGenerator, StyleGAN2Discriminator
from losses import GANLoss, HEARLoss, IDLoss, PerceptualLoss, WFMLoss, g_path_regularize, r1_penalty
from misc.model import FaceParsing
from misc.tools import tensor2cv_8uc3_bgr

DEVICE_TYPE = "cuda"

if DEVICE_TYPE == "cuda":
    assert torch.cuda.is_available()


@dataclass
class TrainSet:

    device: torch.device = torch.device(DEVICE_TYPE, 0)
    is_distributed: bool = False

    base_log_path: str = "log/train"
    ckpt_dir: str = "ckpt"
    sample_dir: str = "sample"
    tb_log_dir: str = "tb_log"

    save_samples_every: int = 250
    save_ckpt_every: int = 5000
    grad_accum_steps: int = 1

    ckpt: None | str = None
    batch_size: int = 8
    size: int = 512
    style_dim: int = 512
    n_mlp: int = 8
    isconcat: bool = False
    noise_blank_level: int = 9

    d_lr: float = 0.0015
    g_lr: float = 0.0015

    r1_reg_every: int = 16
    r1_reg_weight: float = 10.0

    enable_path_reg: bool = True
    path_reg_every: int = 64
    path_reg_weight: float = 0.1

    gan_type: str = "wgan_softplus"
    gan_loss_weight: float = 1.0

    id_loss_weight: float = 1.0

    wfm_feat_index: dict[int, float] = field(default_factory=lambda: {4: 0.5, 5: 0.5})
    wfm_weight: float = 0.25

    enable_hear: bool = False
    hear_weight: float = 1.0
    same_prob: float = 0.0

    enable_perceptual_loss: bool = False
    perceptual_loss_type: str = "vgg19"
    perceptual_loss_layer_weights: dict[str, float] = field(default_factory=lambda: {"conv3_4": 1.0 / 2, "conv4_3": 1.0 / 1})
    perceptual_loss_use_input_norm: bool = True
    perceptual_loss_range_norm: bool = True
    perceptual_loss_perceptual_weight: float = 0.0
    perceptual_loss_style_weight: float = 0.0
    perceptual_loss_criterion: str = "l1"

    mask_loss_weight: float = 1.0

    # "/opt/share/deepfake/dataset_1/ffhq_1024",
    # "/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024",
    # "/opt/share/deepfake/dataset_1/vggface2_hq512",
    # "/opt/share/deepfake/dataset_1/closed_eyes"
    # "/opt/share/deepfake/dataset_1/large_angle"
    # "/opt/share/deepfake/all_dataSet/男女混合/aligned" 48170
    # "/opt/share/deepfake/all_dataSet/男" 5350
    # "/opt/share/deepfake/all_dataSet/女/aligned"
    # "/opt/share/deepfake/all_dataSet/黄种人-FFHQ"
    # "/opt/share/deepfake/all_dataSet/baidu爬取男性45度侧脸及各种风格脸型dst(已经切脸，没有mask)"
    # "/opt/share/deepfake/all_dataSet/generated_yellow-stylegan2"
    # "/opt/share/deepfake/lsy-close-eyes" 6042
    # "/opt/share/deepfake/stars/bcjz/aligned-clean-all"
    # "/opt/share/deepfake/stars/hsx"
    # "/opt/share/deepfake/wang_train_data2/大侧脸/renwu_dace"
    # "/opt/share/deepfake/wang_train_data2/大侧脸/刘亦菲侧脸2"

    target_folders: list[tuple[str, float]] = field(
        default_factory=lambda: [
            ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
            ("/opt/share/deepfake/dataset_1/youtube/What_s_considered_tall_in_South_Korea_Street_Interview_align_results", 0.0),
            ("/opt/share/deepfake/dataset_1/youtube/4k_Face_Close_Up_HDR_Video_Vivid_Colors_Ambient_Sound_-_Relaxing_align_results", 0.0),
        ]
    )
    source_folders: list[tuple[str, float]] = field(
        default_factory=lambda: [
            ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ]
    )

    dataset_rando_sample: bool = True

    def print_config(self):
        print("Training Configuration:")

        cfg_dict = asdict(self)

        max_key_len = max(len(k) for k in cfg_dict.keys())

        for key in sorted(cfg_dict.keys()):
            value = cfg_dict[key]
            if isinstance(value, list) and key.endswith("folders"):
                print(f"  {key:<{max_key_len}} = {len(value)} folders")
            elif isinstance(value, dict):
                print(f"  {key:<{max_key_len}} = {{")
                max_sub_key_len = max(len(str(k)) for k in value) if value else 0
                for sub_key, sub_value in value.items():
                    print(f"  {'':<{max_key_len}}     {sub_key:<{max_sub_key_len}} = {sub_value}")
                print(f"  {'':<{max_key_len}} }}")
            else:
                print(f"  {key:<{max_key_len}} = {value}")


class Trainer:
    def __init__(self, cfg: TrainSet):

        torch.cuda.set_device(cfg.device)

        self.cfg = cfg
        self.iter = 1
        self.loss_dict = {}

        base_log_path = Path(cfg.base_log_path)

        self.ckpt_dir = base_log_path / cfg.ckpt_dir
        self.sample_dir = base_log_path / cfg.sample_dir
        self.tb_log_dir = base_log_path / cfg.tb_log_dir

        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.sample_dir.mkdir(exist_ok=True, parents=True)
        self.tb_log_dir.mkdir(exist_ok=True, parents=True)

        if cfg.enable_path_reg:
            self.mean_path_length = 0

        if cfg.ckpt is None:
            checkpoint = None
        else:
            print(f"Loading checkpoint from {cfg.ckpt}")
            checkpoint = torch.load(cfg.ckpt, map_location=torch.device("cpu"), weights_only=False)

        # ------------------------ init model ------------------------
        self.net_d = StyleGAN2Discriminator(out_size=cfg.size).to(cfg.device)
        self.net_g = FullGenerator(size=cfg.size, style_dim=cfg.style_dim, n_mlp=cfg.n_mlp, isconcat=cfg.isconcat).to(cfg.device)
        self.net_g_ema = FullGenerator(size=cfg.size, style_dim=cfg.style_dim, n_mlp=cfg.n_mlp, isconcat=cfg.isconcat).to(cfg.device)

        if checkpoint is not None:

            if "config" in checkpoint:
                self._ckpt_train_cfg = TrainSet(**checkpoint["config"])

            if "iter" in checkpoint:
                self.iter = checkpoint["iter"]
                print(f"iter = {self.iter}")

            if "d" in checkpoint:
                self.net_d.load_state_dict(checkpoint["d"])
                print("Loaded net_d state from checkpoint.")

            if "g" in checkpoint:
                self.net_g.load_state_dict(checkpoint["g"])
                print("Loaded net_g state from checkpoint.")

            if "g_ema" in checkpoint:
                self.net_g_ema.load_state_dict(checkpoint["g_ema"])
                print("Loaded net_g_ema state from checkpoint.")

        self.net_d.train()
        self.net_g.train()
        self.net_g_ema.eval()

        if cfg.is_distributed:
            self.net_d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_d)
            self.net_d = DDP(self.net_d, device_ids=[cfg.device.index], output_device=cfg.device.index, broadcast_buffers=False)
            self.net_g = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_g)
            self.net_g = DDP(self.net_g, device_ids=[cfg.device.index], output_device=cfg.device.index, broadcast_buffers=True)

        # ------------------------ init optimizer ------------------------
        net_d_params = self.net_d.module.parameters() if cfg.is_distributed else self.net_d.parameters()
        net_g_params = self.net_g.module.parameters() if cfg.is_distributed else self.net_g.parameters()

        self.d_optimizer = torch.optim.Adam(params=net_d_params, lr=cfg.d_lr, betas=(0.0, 0.99))
        self.g_optimizer = torch.optim.Adam(params=net_g_params, lr=cfg.g_lr, betas=(0.0, 0.99))

        if checkpoint is not None:
            if "d_optim" in checkpoint:
                self.d_optimizer.load_state_dict(checkpoint["d_optim"])
                for param_group in self.d_optimizer.param_groups:
                    param_group["lr"] = cfg.d_lr
                print("Loaded d_optimizer state from checkpoint.")

            if "g_optim" in checkpoint:
                self.g_optimizer.load_state_dict(checkpoint["g_optim"])
                for param_group in self.g_optimizer.param_groups:
                    param_group["lr"] = cfg.g_lr
                print("Loaded g_optimizer state from checkpoint.")

        # ------------------------ init AMP GradScaler ------------------------
        self.scaler_d = GradScaler()
        self.scaler_g = GradScaler()

        if checkpoint is not None:
            if "scaler_d" in checkpoint:
                self.scaler_d.load_state_dict(checkpoint["scaler_d"])
                print("Loaded scaler_d state from checkpoint.")

            if "scaler_g" in checkpoint:
                self.scaler_g.load_state_dict(checkpoint["scaler_g"])
                print("Loaded scaler_g state from checkpoint.")

        # ------------------------ cfg loss ------------------------

        self.gan_loss = GANLoss(cfg.gan_type, loss_weight=cfg.gan_loss_weight).to(cfg.device)
        self.id_loss = IDLoss(weight=cfg.id_loss_weight).to(device=cfg.device)
        self.wfm_loss = WFMLoss(cfg.wfm_feat_index, cfg.wfm_weight).to(cfg.device)

        if cfg.enable_perceptual_loss:
            self.perceptual_loss = PerceptualLoss(
                cfg.perceptual_loss_layer_weights,
                cfg.perceptual_loss_type,
                cfg.perceptual_loss_use_input_norm,
                cfg.perceptual_loss_range_norm,
                cfg.perceptual_loss_perceptual_weight,
                cfg.perceptual_loss_style_weight,
                cfg.perceptual_loss_criterion,
            ).to(cfg.device)

        if cfg.enable_hear:
            self.hear_loss = HEARLoss(cfg.hear_weight).to(cfg.device)

        # ------------------------ init dataset_loader ------------------------
        pipe = datasetloader(
            batch_size=cfg.batch_size,
            num_threads=2,
            prefetch_queue_depth=4,
            py_num_workers=2,
            py_start_method="spawn",
            device_id=cfg.device.index,
            resize=cfg.size,
            target_folders=cfg.target_folders,
            source_folders=cfg.source_folders,
            same_image_prob=cfg.same_prob,
            rando=cfg.dataset_rando_sample,
        )

        self.loader = DALIGenericIterator(pipelines=pipe, output_map=["tgt", "src", "is_same"], auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

        # ------------------------ log ------------------------
        if cfg.device.index == 0:
            self.writer = SummaryWriter(log_dir=self.tb_log_dir)

        # ------------------------ face utils ------------------------
        self.face_pras = FaceParsing.Model().to(device=cfg.device)

        # ------------------------------------------------
        if self.cfg.device.index == 0:
            self.cfg.print_config()

    def model_ema(self, decay=0.999):

        net_g_params = self.net_g.module if cfg.is_distributed else self.net_g
        net_g_params = dict(net_g_params.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def save(self):

        if self.cfg.is_distributed:
            d = self.net_d.module
            g = self.net_g.module
        else:
            d = self.net_d
            g = self.net_g

        train_info = asdict(self.cfg)

        train_info.pop("target_folders", None)
        train_info.pop("source_folders", None)

        state_dict = {
            "iter": self.iter,
            "d": d.state_dict(),
            "g": g.state_dict(),
            "g_ema": self.net_g_ema.state_dict(),
            "d_optim": self.d_optimizer.state_dict(),
            "g_optim": self.g_optimizer.state_dict(),
            "scaler_d": self.scaler_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
            "train_info": train_info,
        }

        ckpt_file = self.ckpt_dir / f"{self.iter}_{self.cfg.size}.pth"
        torch.save(state_dict, ckpt_file)

    def fetch_training_batch(self):

        data = self.loader.next()[0]

        target, source, same = data["tgt"], data["src"], data["is_same"]

        target_mask = self.face_pras.process(target)
        source_mask = self.face_pras.process(source)

        return target, target_mask, source, source_mask, same.squeeze()

    def reduce_loss_dict(self):

        if not self.cfg.is_distributed:
            return

        world_size = dist.get_world_size()

        if world_size < 2:
            return

        with torch.no_grad():

            sorted_items = sorted(self.loss_dict.items())
            keys, losses = zip(*sorted_items) if sorted_items else ([], [])

            losses_tensor = torch.stack(losses, dim=0)
            dist.reduce(losses_tensor, op=dist.ReduceOp.SUM, dst=0)
            if self.cfg.device.index == 0:
                losses_tensor /= world_size
                self.loss_dict.update(zip(keys, losses_tensor))

    @torch.no_grad()
    def save_sample(self):

        target, target_mask, source, source_mask, _ = self.fetch_training_batch()
        z_id = self.id_loss.extract_id_feats(source, source_mask)

        z_id = torch.zeros_like(z_id, device=z_id.device, dtype=z_id.dtype)

        fake, _, fake_mask = self.net_g_ema(target, z_id, noise_blank_level=self.cfg.noise_blank_level)

        fake = fake * fake_mask + (1 - fake_mask) * target
        fake_mask = fake_mask.sub_(0.5).mul_(2.0).expand(-1, 3, -1, -1)
        target_mask = target_mask.sub_(0.5).mul_(2.0).expand(-1, 3, -1, -1)

        grid = torch.cat((target, source, fake, fake_mask, target_mask), 0)

        grid = make_grid(grid, nrow=self.cfg.batch_size)

        grid_cpu = tensor2cv_8uc3_bgr(grid, value_range=(-1, 1), swap_rb_ch=True)

        sample_file = self.sample_dir / f"{self.iter}_{self.cfg.size}.png"

        cv2.imwrite(sample_file, grid_cpu.squeeze(0), [cv2.IMWRITE_PNG_COMPRESSION, 3])

    def train_loop(self):

        cfg = self.cfg
        count = itertools.count(start=self.iter)
        if cfg.device.index == 0:
            pbar = tqdm(count, initial=self.iter, unit="iter")
        else:
            pbar = count

        torch.cuda.empty_cache()
        for self.iter in pbar:

            # =============== 训练判别器 ===============
            self.net_d.requires_grad_(True)
            self.net_g.requires_grad_(False)
            self.d_optimizer.zero_grad()

            with autocast(device_type=DEVICE_TYPE, dtype=torch.float16):

                for accum_step in range(cfg.grad_accum_steps):

                    target, target_mask, source, source_mask, same = self.fetch_training_batch()
                    is_r1_reg_every = (self.iter % cfg.r1_reg_every == 0) and (accum_step == cfg.grad_accum_steps - 1)

                    with torch.no_grad():
                        z_id = self.id_loss.extract_id_feats(source, source_mask)
                        fake_img, _, _ = self.net_g(target, z_id, noise_blank_level=cfg.noise_blank_level)

                    fake_pred, _ = self.net_d(fake_img)

                    if is_r1_reg_every:
                        target.requires_grad_(True)

                    real_pred, _ = self.net_d(target)

                    d_loss = self.gan_loss(real_pred, True, is_disc=True) + self.gan_loss(fake_pred, False, is_disc=True)

                    self.loss_dict["D"] = d_loss.detach()
                    self.loss_dict["real_score"] = real_pred.detach()
                    self.loss_dict["fake_score"] = fake_pred.detach()

                    d_loss = d_loss / cfg.grad_accum_steps

                    if is_r1_reg_every:
                        r1 = r1_penalty(real_pred, target)
                        r1 = cfg.r1_reg_weight / 2 * r1 * cfg.r1_reg_every
                        d_loss += r1
                        self.loss_dict["r1"] = r1.detach()

                    self.scaler_d.scale(d_loss).backward()

            self.scaler_d.step(self.d_optimizer)
            self.scaler_d.update()

            # =============== 训练生成器 ===============
            self.net_d.requires_grad_(False)
            self.net_g.requires_grad_(True)
            self.g_optimizer.zero_grad()

            with autocast(device_type=DEVICE_TYPE, dtype=torch.float16):

                for accum_step in range(cfg.grad_accum_steps):

                    target, target_mask, source, source_mask, same = self.fetch_training_batch()
                    with torch.no_grad():
                        z_id = self.id_loss.extract_id_feats(source, source_mask)

                    fake_img, latents, fake_mask = self.net_g(target, z_id, return_latents=True, noise_blank_level=cfg.noise_blank_level)

                    # ------------- 对抗损失 -------------
                    fake_pred, fake_feats = self.net_d(fake_img)
                    g_loss = self.gan_loss(fake_pred, True, is_disc=False)
                    self.loss_dict["G"] = g_loss.detach()

                    # ------------- 身份损失 -------------
                    fake_z_id = self.id_loss.extract_id_feats(fake_img)
                    id_loss = self.id_loss(fake_z_id, z_id)
                    g_loss += id_loss
                    self.loss_dict["ID"] = id_loss.detach() / cfg.id_loss_weight

                    # --------------WFM---------------
                    with torch.no_grad():
                        real_feats = self.net_d.get_feats(target)
                    wfm_loss = self.wfm_loss(fake_feats, real_feats)
                    g_loss += wfm_loss
                    self.loss_dict["wfm"] = wfm_loss.detach()

                    # ------------- 感知损失 -------------
                    if cfg.enable_perceptual_loss:
                        percep_loss, style_loss = self.perceptual_loss(fake_img, target)
                        if percep_loss is not None:
                            g_loss += percep_loss
                            self.loss_dict["Percep"] = percep_loss.detach()
                        if style_loss is not None:
                            g_loss += style_loss
                            self.loss_dict["Style"] = style_loss.detach()

                    # ------------- 启发式误差识别细化(FaceShifter HEAR) -------------
                    if cfg.enable_hear:
                        rec_pix = self.hear_loss(fake_img, target, same)
                        g_loss += rec_pix
                        self.loss_dict["Pixel_Rec"] = rec_pix.detach()

                    # ------------- 掩码损失 -------------
                    with autocast(device_type=DEVICE_TYPE, enabled=False):
                        mask_loss = F.binary_cross_entropy(fake_mask, target_mask, reduction="mean")
                        g_loss += mask_loss * cfg.mask_loss_weight
                        self.loss_dict["Mask"] = mask_loss.detach()

                    # -----------------------------------
                    g_loss = g_loss / cfg.grad_accum_steps

                    # 路径正则化
                    if cfg.enable_path_reg:
                        if (self.iter % cfg.path_reg_every == 0) and (accum_step == cfg.grad_accum_steps - 1):
                            path_loss, self.mean_path_length, path_lengths = g_path_regularize(fake_img, latents, self.mean_path_length)
                            weighted_path_loss = cfg.path_reg_weight * cfg.path_reg_every * path_loss
                            g_loss += weighted_path_loss
                            self.loss_dict["path_loss"] = path_loss.detach()
                            self.loss_dict["path_lengths"] = path_lengths.detach()
                            self.loss_dict["mean_path_length"] = self.mean_path_length.detach()

                    self.scaler_g.scale(g_loss).backward()

            self.scaler_g.step(self.g_optimizer)
            self.scaler_g.update()

            if cfg.device.index == 0:
                self.model_ema()

            if self.iter % 10 == 0:
                self.reduce_loss_dict()

            if self.iter % 1500 == 0:
                torch.cuda.empty_cache()

            if cfg.device.index == 0:
                for k, v in self.loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        v = v.mean().item()

                    self.writer.add_scalar(f"Loss/{k}", v, self.iter)

                if self.iter % cfg.save_samples_every == 0:
                    self.save_sample()

                if self.iter % cfg.save_ckpt_every == 0:
                    self.save()


def train(cfg: TrainSet):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"多卡训练: world_size={world_size}, local_rank={local_rank}")
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size)
        cfg.device = torch.device(DEVICE_TYPE, local_rank)
        cfg.is_distributed = is_distributed
    else:
        print("单卡训练")
        cfg.is_distributed = is_distributed

    trainer = Trainer(cfg)
    try:
        trainer.train_loop()
    finally:
        if trainer.cfg.is_distributed:
            torch.distributed.barrier()
        if trainer.cfg.device.index == 0:
            trainer.save()
            trainer.writer.close()
            dist.destroy_process_group()

        torch.cuda.empty_cache()


if __name__ == "__main__":

    # cfg_4 = TrainSet()
    # cfg_4.base_log_path = "log/train_4"
    # cfg_4.ckpt = "log/train_4/ckpt/305545_512.pth"
    # cfg_4.batch_size = 8
    # cfg_4.d_lr = 0.0007
    # cfg_4.g_lr = 0.0007
    # cfg_4.enable_path_reg = True
    # cfg_4.enable_hear = False
    # cfg_4.noise_blank_level = 9
    # cfg = cfg_4

    # cfg_5 = TrainSet()
    # cfg_5.base_log_path = "log/train_5"
    # cfg_5.ckpt = "log/train_5/ckpt/205692_512.pth"
    # cfg_5.batch_size = 16
    # cfg_5.d_lr = 0.0002
    # cfg_5.g_lr = 0.0002
    # cfg_5.enable_path_reg = True
    # cfg_5.enable_hear = False
    # cfg_5.noise_blank_level = 5
    # cfg = cfg_5

    cfg_6 = TrainSet()
    cfg_6.base_log_path = "log/train_6"
    # cfg_6.ckpt = "log/train_5/ckpt/205692_512.pth"
    cfg_6.batch_size = 8
    cfg_6.d_lr = 0.0015
    cfg_6.g_lr = 0.0015
    cfg_6.enable_path_reg = True
    cfg_6.noise_blank_level = 5
    cfg_6.id_loss_weight = 1.0
    cfg_6.enable_hear = False
    cfg_6.enable_perceptual_loss = False
    cfg = cfg_6

    train(cfg)
