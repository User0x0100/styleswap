import itertools
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
import torch.distributed as dist
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader import datasetloader
from model import FullGenerator, StyleGAN2Discriminator
from losses import GANLoss, IDLoss, PerceptualLoss, WFMLoss, g_path_regularize, r1_penalty
from misc.model import FaceParsing, IDEncoder
from misc.tools import tensor2cv_8uc3_bgr


if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    raise RuntimeError("CUDA is not available. Exiting.")


AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class TrainSet:

    device: torch.device = torch.device(DEVICE_TYPE, 0)
    is_distributed: bool = False

    enable_amp: bool = False

    base_log_path: str = "log/train"
    ckpt_dir: str = "ckpt"
    sample_dir: str = "sample"
    tb_log_dir: str = "tb_log"

    save_samples_every: int = 250
    save_ckpt_every: int = 5000
    grad_accum_steps: int = 1

    ckpt: None | str = None
    batch_size: int = 12
    size: int = 512
    style_dim: int = 512
    n_mlp: int = 8
    isconcat: bool = True
    noise_blank_level: int = 9

    d_lr: float = 0.0001
    g_lr: float = 0.0001

    r1_reg_every: int = 16
    r1_reg_weight: float = 10.0

    enable_path_reg: bool = False
    path_reg_every: int = 64
    path_reg_weight: float = 0.1

    # 对抗损失
    gan_type: str = "wgan_softplus"
    gan_loss_weight: float = 1.0

    # 身份损失
    id_encode_provider: IDEncoder.PROVIDER = IDEncoder.PROVIDER.ARC_OFFICIAL_R50
    id_loss_weight: float = 10.0

    # 弱特征匹配
    enable_wfm: bool = True
    wfm_feat_index: dict[int, float] = field(default_factory=lambda: {6: 1.0})
    wfm_weight: float = 100.0

    # enable_hear: bool = False
    # hear_weight: float = 1.0
    # same_prob: float = 0.1

    # 重建损失
    enable_perceptual_loss: bool = False
    perceptual_loss_type: str = "vgg19"
    perceptual_loss_layer_weights: dict[str, float] = field(
        default_factory=lambda: {
            "conv3_4": 1.0 / 2.0,
            "conv4_3": 1.0 / 2.0,
        }
    )
    perceptual_loss_use_input_norm: bool = True
    perceptual_loss_range_norm: bool = True
    perceptual_loss_perceptual_weight: float = 100.0
    perceptual_loss_style_weight: float = 0.0
    perceptual_loss_criterion: str = "l1"

    # 掩码损失
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
            ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
            # ("/opt/share/deepfake/dataset_1/large_angle", -1.0),
            ("/opt/share/deepfake/dataset_1/RealOcc/image/realign_arcface_dst", 2.0),
        ]
    )
    source_folders: list[tuple[str, float]] = field(
        default_factory=lambda: [
            ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
            # ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
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
        # torch.backends.cudnn.benchmark = True
        # torch.set_float32_matmul_precision("high")
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True
        print(f"AMP: {cfg.enable_amp}")
        if cfg.enable_amp:
            print(f"AMP_DTYPE: {AMP_DTYPE}")

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

        if cfg.device.index == 0:
            mp.set_start_method("spawn", force=True)
            self.save_sample_queue = mp.Queue(maxsize=10)
            self.save_sample_process = mp.Process(target=self.save_sample_worker, name="save_sample_worker", args=(self.save_sample_queue,), daemon=True)
            self.save_sample_process.start()

        if cfg.ckpt is None:
            checkpoint = None
        else:
            print(f"Loading checkpoint from {cfg.ckpt}")
            checkpoint = torch.load(cfg.ckpt, map_location=torch.device("cpu"), weights_only=False)

        if cfg.enable_path_reg:
            self.mean_path_length = 0.0
            if checkpoint is not None and "mean_path_length" in checkpoint:
                self.mean_path_length = checkpoint["mean_path_length"]

        self.imgs_per_update = cfg.batch_size * cfg.grad_accum_steps * (dist.get_world_size() if cfg.is_distributed else 1)

        # ------------------------ init model ------------------------
        self.net_d = StyleGAN2Discriminator(out_size=cfg.size).to(cfg.device)
        self.net_g = FullGenerator(size=cfg.size, style_dim=cfg.style_dim, n_mlp=cfg.n_mlp, isconcat=cfg.isconcat).to(cfg.device)
        self.net_g_ema = FullGenerator(size=cfg.size, style_dim=cfg.style_dim, n_mlp=cfg.n_mlp, isconcat=cfg.isconcat).to(cfg.device)

        for name, buf in self.net_d.named_buffers():
            if buf.requires_grad:
                print(f"Warning: Buffer '{name}' in net_d requires grad, You need to pay attention to the implementation of EMA.")

        for name, buf in self.net_g.named_buffers():
            if buf.requires_grad:
                print(f"Warning: Buffer '{name}' in net_g requires grad, You need to pay attention to the implementation of EMA.")

        if checkpoint is not None:

            if "train_info" in checkpoint:
                self._ckpt_train_info = TrainSet(**checkpoint["train_info"])

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
        self.net_g_ema.requires_grad_(False)

        if cfg.is_distributed:
            self.net_d = DDP(self.net_d, device_ids=[cfg.device.index], output_device=cfg.device.index, broadcast_buffers=False)
            self.net_g = DDP(self.net_g, device_ids=[cfg.device.index], output_device=cfg.device.index, broadcast_buffers=False)

        # ------------------------ init optimizer ------------------------
        net_d_params = self.net_d.module.parameters() if cfg.is_distributed else self.net_d.parameters()
        net_g_params = self.net_g.module.parameters() if cfg.is_distributed else self.net_g.parameters()

        enable_optimizer_fused = True if AMP_DTYPE != torch.float16 else False
        print(f"enable_optimizer_fused: {enable_optimizer_fused}")
        self.d_optimizer = torch.optim.Adam(params=net_d_params, lr=cfg.d_lr, betas=(0.0, 0.99), fused=enable_optimizer_fused)
        self.g_optimizer = torch.optim.Adam(params=net_g_params, lr=cfg.g_lr, betas=(0.0, 0.99), fused=enable_optimizer_fused)

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
        # bf16 不需要 GradScaler fp16 需要
        enable_scaler = cfg.enable_amp and AMP_DTYPE == torch.float16
        print(f"enable_scaler: {enable_scaler}")
        self.scaler_d = GradScaler(enabled=enable_scaler)
        self.scaler_g = GradScaler(enabled=enable_scaler)

        if checkpoint is not None:
            if "scaler_d" in checkpoint:
                self.scaler_d.load_state_dict(checkpoint["scaler_d"])
                print("Loaded scaler_d state from checkpoint.")

            if "scaler_g" in checkpoint:
                self.scaler_g.load_state_dict(checkpoint["scaler_g"])
                print("Loaded scaler_g state from checkpoint.")

        # ------------------------ cfg loss ------------------------

        self.gan_loss = GANLoss(cfg.gan_type, loss_weight=cfg.gan_loss_weight).to(cfg.device)
        self.id_loss = IDLoss(weight=cfg.id_loss_weight, id_encode_provider=cfg.id_encode_provider).to(device=cfg.device)

        if cfg.enable_wfm:
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

        # if cfg.enable_hear:
        #     self.hear_loss = HEARLoss(cfg.hear_weight).to(cfg.device)

        # ------------------------ init dataset_loader ------------------------
        pipe = datasetloader(
            batch_size=cfg.batch_size,
            num_threads=2,
            prefetch_queue_depth=4,
            py_num_workers=2,
            py_start_method="spawn",
            device_id=cfg.device.index,
            resize=cfg.size,
            src=cfg.target_folders,
            source_folders=cfg.source_folders,
            same_image_prob=0.0,
            random_sampling=cfg.dataset_rando_sample,
        )

        self.dataset = DALIGenericIterator(pipelines=pipe, output_map=["tgt", "src", "is_same"], auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

        # ------------------------ log ------------------------
        if cfg.device.index == 0:
            self.writer = SummaryWriter(log_dir=self.tb_log_dir)

        # ------------------------ face utils ------------------------
        self.face_pras = FaceParsing.Model(occ=True).to(device=cfg.device)

        # ------------------------------------------------
        if self.cfg.device.index == 0:
            self.cfg.print_config()

    @torch.no_grad()
    def model_ema(self, decay=0.999):

        alpha = 1 - decay
        net_g = self.net_g.module if self.cfg.is_distributed else self.net_g

        for (name, param), (name_ema, param_ema) in zip(net_g.named_parameters(), self.net_g_ema.named_parameters()):
            assert name == name_ema
            param_ema.lerp_(param, weight=alpha)

        for (name_buf, buf), (name_buf_ema, buf_ema) in zip(net_g.named_buffers(), self.net_g_ema.named_buffers()):
            assert name_buf == name_buf_ema
            buf_ema.copy_(buf)

    def save_state_dict(self):

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

        if hasattr(self, "mean_path_length"):
            if isinstance(self.mean_path_length, Tensor):
                mean_path_length = self.mean_path_length.item()
            else:
                mean_path_length = self.mean_path_length

            state_dict["mean_path_length"] = mean_path_length

        ckpt_file = self.ckpt_dir / f"{self.iter}.pth"
        torch.save(state_dict, ckpt_file)

    @torch.no_grad()
    def fetch_training_batch(self, fetch_num=1) -> List[Tuple[Tensor, ...]]:

        samples: List[Tuple[Tensor, ...]] = []

        for _ in range(fetch_num):
            data = self.dataset.next()[0]

            target, source, same = data["tgt"], data["src"], data["is_same"]

            target_mask = self.face_pras.forward(target)
            # source_mask = self.face_pras.process(source)

            samples.append((target, target_mask, source))

        return samples

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

    @staticmethod
    def save_sample_worker(q: mp.Queue):

        while True:
            source, target, target_mask, fake, fake_mask, sp = q.get(block=True)

            B = target.shape[0]

            grid = torch.cat((target, source, fake, fake_mask, target_mask), 0)
            grid = make_grid(grid, nrow=B)
            grid_cpu = tensor2cv_8uc3_bgr(grid, value_range=(-1, 1), swap_rb_ch=True)

            cv2.imwrite(sp, grid_cpu.squeeze(0), [cv2.IMWRITE_PNG_COMPRESSION, 3])

    def train_loop(self):

        cfg = self.cfg
        count = itertools.count(start=self.iter)
        if cfg.device.index == 0:
            pbar = tqdm(count, initial=self.iter, unit="iter")
        else:
            pbar = count

        ema_decay = 0.5 ** (self.imgs_per_update / (10.0 * 1000))

        for self.iter in pbar:

            with autocast(device_type=DEVICE_TYPE, dtype=AMP_DTYPE, enabled=cfg.enable_amp):

                train_samples = self.fetch_training_batch(cfg.grad_accum_steps)
                cached_ids: List[Tensor] = []

                # =============== 训练判别器 ===============
                self.net_d.requires_grad_(True)
                self.net_g.requires_grad_(False)
                self.d_optimizer.zero_grad()

                for accum_step, sample in enumerate(train_samples):

                    target, target_mask, source = sample
                    is_r1_reg_every = (self.iter % cfg.r1_reg_every == 0) and (accum_step == cfg.grad_accum_steps - 1)

                    with torch.no_grad():
                        source_id = self.id_loss.get_id_feats(source, None, realign=False)
                        cached_ids.append(source_id)
                        fake_img, _, _ = self.net_g(target, source_id, noise_blank_level=cfg.noise_blank_level)

                    fake_pred, _ = self.net_d(fake_img)

                    if is_r1_reg_every:
                        target.requires_grad_(True)

                    real_pred, _ = self.net_d(target)

                    d_loss: Tensor = self.gan_loss(real_pred, True, is_disc=True) + self.gan_loss(fake_pred, False, is_disc=True)

                    self.loss_dict["D"] = d_loss.detach()
                    self.loss_dict["real_score"] = real_pred.detach()
                    self.loss_dict["fake_score"] = fake_pred.detach()

                    d_loss /= cfg.grad_accum_steps

                    if is_r1_reg_every:
                        with autocast(device_type=DEVICE_TYPE, enabled=False):
                            r1 = r1_penalty(real_pred, target)
                            r1 = cfg.r1_reg_weight / 2 * r1 * cfg.r1_reg_every
                        d_loss += r1
                        self.loss_dict["r1"] = r1.detach()
                        target.requires_grad_(False)

                    self.scaler_d.scale(d_loss).backward()

                self.scaler_d.step(self.d_optimizer)
                self.scaler_d.update()

                # =============== 训练生成器 ===============
                self.net_d.requires_grad_(False)
                self.net_g.requires_grad_(True)
                self.g_optimizer.zero_grad()

                for accum_step, sample in enumerate(train_samples):

                    target, target_mask, source = sample
                    with torch.no_grad():
                        source_id = cached_ids[accum_step]

                    fake_img, latents, fake_mask = self.net_g(target, source_id, return_latents=True, noise_blank_level=cfg.noise_blank_level)

                    # ------------- 对抗损失 -------------
                    fake_pred, fake_feats = self.net_d(fake_img)
                    g_loss: Tensor = self.gan_loss(fake_pred, True, is_disc=False)
                    self.loss_dict["G"] = g_loss.detach()

                    # ------------- 身份损失 -------------
                    fake_id = self.id_loss.get_id_feats(fake_img, None, realign=False)
                    id_loss = self.id_loss(fake_id, source_id)
                    g_loss += id_loss
                    self.loss_dict["ID"] = id_loss.detach() / cfg.id_loss_weight

                    # --------------WFM---------------
                    if cfg.enable_wfm:
                        with torch.no_grad():
                            real_feats = self.net_d.feature_maps(target)
                        wfm_loss = self.wfm_loss(fake_feats, real_feats)
                        g_loss += wfm_loss
                        self.loss_dict["wfm"] = wfm_loss.detach()

                    # ------------- 重建损失 -------------
                    if cfg.enable_perceptual_loss:
                        percep_loss, style_loss = self.perceptual_loss(fake_img, target)
                        if percep_loss is not None:
                            g_loss += percep_loss
                            self.loss_dict["Percep"] = percep_loss.detach()
                        if style_loss is not None:
                            g_loss += style_loss
                            self.loss_dict["Style"] = style_loss.detach()

                    # ------------- 启发式误差识别细化(FaceShifter HEAR) -------------
                    # if cfg.enable_hear:
                    #     rec_pix = self.hear_loss(fake_img, target, same, mask=target_mask, p=cfg.same_prob)
                    #     g_loss += rec_pix
                    #     self.loss_dict["Pixel_Rec"] = rec_pix.detach()

                    # ------------- 掩码损失 -------------
                    with autocast(device_type=DEVICE_TYPE, enabled=False):
                        mask_loss = F.binary_cross_entropy(fake_mask, target_mask, reduction="mean")
                        g_loss += mask_loss * cfg.mask_loss_weight
                        self.loss_dict["Mask"] = mask_loss.detach()

                    # -----------------------------------
                    g_loss /= cfg.grad_accum_steps

                    # 路径正则化
                    if cfg.enable_path_reg:
                        if (self.iter % cfg.path_reg_every == 0) and (accum_step == cfg.grad_accum_steps - 1):
                            with autocast(device_type=DEVICE_TYPE, enabled=False):
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

                self.model_ema(decay=ema_decay)

                if self.iter > 100:
                    for k, v in self.loss_dict.items():
                        if isinstance(v, Tensor):
                            v = v.mean().item()
                        self.writer.add_scalar(f"Loss/{k}", v, self.iter)

                if self.iter % cfg.save_samples_every == 0:

                    with torch.no_grad():
                        fake, _, fake_mask = self.net_g_ema(target, source_id, noise_blank_level=cfg.noise_blank_level)

                        # fake = fake * fake_mask + (1 - fake_mask) * target
                        fake_mask = fake_mask.sub_(0.5).mul_(2.0).expand(-1, 3, -1, -1)
                        target_mask = target_mask.sub_(0.5).mul_(2.0).expand(-1, 3, -1, -1)

                    sp = self.sample_dir / f"{self.iter}.png"
                    source_cpu = source.detach().cpu()
                    target_cpu = target.detach().cpu()
                    target_mask_cpu = target_mask.detach().cpu()
                    fake_cpu = fake.detach().cpu()
                    fake_mask_cpu = fake_mask.detach().cpu()

                    self.save_sample_queue.put((source_cpu, target_cpu, target_mask_cpu, fake_cpu, fake_mask_cpu, sp), block=False)

                if self.iter % cfg.save_ckpt_every == 0:
                    self.save_state_dict()


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
        if trainer.cfg.device.index == 0:
            trainer.save_state_dict()
            trainer.writer.close()
            trainer.save_sample_process.terminate()
            trainer.save_sample_process.join(timeout=5)

        if trainer.cfg.is_distributed:
            torch.distributed.barrier()
            dist.destroy_process_group()

        torch.cuda.empty_cache()


if __name__ == "__main__":

    cfg = TrainSet()
    train(cfg)
