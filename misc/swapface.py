import multiprocessing as mp
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch import autocast, nn
from torch.nn import functional as F
from torchcodec.decoders import VideoDecoder
from torchvision.io import decode_image
from tqdm import tqdm

from face_swap_model import FullGenerator
from losses.id_loss import IDLoss
from misc.model import FaceParsing

from .model import IDEncoder
from .tools import FaceAlign, tensor2cv_8uc3_bgr
from .utils import DataFolder


class Swapper(nn.Module):
    G_STATE_DICT = "g_ema"

    def __init__(self, ckpt: str, size: int = 512, isconcat: bool = False, align_dst: None | torch.Tensor = None):
        super().__init__()

        checkpoint = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)

        if not self.G_STATE_DICT in checkpoint:
            available_keys = list(checkpoint.keys())
            raise KeyError(f"{self.G_STATE_DICT} not found in checkpoint. Available keys: {available_keys}")

        self.noise_blank_level = 9
        if "train_info" in checkpoint:
            isconcat = checkpoint["train_info"]["isconcat"]
            self.noise_blank_level = checkpoint["train_info"]["noise_blank_level"]

        self.size = size
        self.swap_net = FullGenerator(size=size, isconcat=isconcat)
        self.swap_net.load_state_dict(checkpoint[self.G_STATE_DICT], strict=False)

        self.swap_net.eval()
        self.swap_net.requires_grad_(False)

        self.aligner = FaceAlign(from_normalized=True)

        self.idencoder = IDLoss(id_encode_provider=IDEncoder.PROVIDER.BLENDFACE)

        if align_dst is None:
            align_dst = IDEncoder.get_dst_pts(size)

        else:
            assert align_dst.shape == [5, 2]

        self.register_buffer("align_dst", align_dst, persistent=False)

        self.facepars = FaceParsing.Model(compile_model=False)

        self.eval()
        self.requires_grad_(False)

    def swap_single(self, aligned_faces: torch.Tensor, id: torch.Tensor) -> torch.Tensor:
        if id.dim() == 1:
            id = id.unsqueeze(0)

        B = aligned_faces.shape[0]
        id = id.expand(B, -1)

        fake_face, _, fake_mask = self.swap_net(aligned_faces, id, noise_blank_level=self.noise_blank_level)
        # fake_face = fake_face * fake_mask + (1 - fake_mask) * aligned_faces
        return fake_face, fake_mask

    def get_fae_id(self, fp: str) -> torch.Tensor:
        device = self.align_dst.device

        image = decode_image(fp)
        image = image.to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0).unsqueeze_(0)
        align_face, _, _ = self.aligner.align(images=image, out_size=(IDEncoder.SIZE,) * 2, dst_pts5_pixel=IDEncoder.get_dst_pts(IDEncoder.SIZE).to(device=device))

        face_mask = self.facepars.process(align_face)
        # align_face = align_face * face_mask + (1.0 - face_mask) * (-1.0)

        id_emb = self.idencoder.extract_id_feats(align_face, face_mask)

        return id_emb

    @autocast(device_type="cuda")
    def swap_folder(self, fp: str, source: str | None = None, wfp: str | None = None, sf_prefix: str = "", batch_size: int = 32):

        device = self.align_dst.device
        data_folder = DataFolder(fp)

        if source is None:
            id_emb = torch.randn((1, 512), dtype=torch.float, device=device)
            id_emb = F.normalize(id_emb, p=2, dim=1)
        else:
            id_emb = self.get_fae_id(source)[0:1]

        if wfp is None:
            wfp = Path(fp).joinpath("swap_results/")

        Path(wfp).mkdir(exist_ok=True, parents=True)

        for images, fns in data_folder.iter_batch_with_tensor(batch_size, True):

            images = [image.float().to(device=device).div_(255.0).mul_(2.0).sub_(1.0) for image in images]
            aligned_faces, norm_affine_matrix, idxs = self.aligner.align(images, (self.size,) * 2, self.align_dst)

            swaped_face, face_mask = self.swap_single(aligned_faces, id_emb[:1])

            fake_id = self.idencoder.extract_id_feats(swaped_face, face_mask)

            similaritys = F.cosine_similarity(fake_id, id_emb.expand(fake_id.shape[0], -1), dim=1)

            for similarity, name in zip(similaritys, fns):
                print(f"相似度: {similarity.mean():.4f} --- {name}")

            restored = FaceAlign.restore_faces_to_original(swaped_face, norm_affine_matrix, images, idxs)

            for swap, fn, org in zip(restored, fns, images):
                name = Path(fn).stem
                sfp = Path(wfp) / f"{sf_prefix}{name}_swaped.png"

                grid = torch.cat((swap, org), dim=2)
                grid = tensor2cv_8uc3_bgr(grid, (-1, 1))

                cv2.imwrite(sfp, grid.squeeze(0), [cv2.IMWRITE_PNG_COMPRESSION, 3])

        print(f"平均相似度为: {similaritys.mean().item():.4f}")

    @staticmethod
    def display_worker(q: mp.Queue, fps=25, value_range=(-1, 1)):

        delay = int(1000 / fps)

        while True:
            frames = q.get()
            if frames is None:
                break
            frames = tensor2cv_8uc3_bgr(frames, value_range)
            for frame in frames:
                cv2.imshow("swaped", frame)
                cv2.waitKey(delay)

    @autocast(device_type="cuda")
    def swap_video(self, vfp: str, source: str | None = None, sfp: str | None = None, batch_size: int = 32):

        device = self.align_dst.device
        if sfp is None:
            fn = Path(vfp).stem
            folder = Path(vfp).parent
            sfp = Path(folder) / f"{fn}_swaped.mp4"

        mp.set_start_method("spawn", force=True)
        q = mp.Queue(maxsize=15)
        p = mp.Process(target=self.display_worker, args=(q,), daemon=True)
        p.start()

        if source is None:
            id_emb = torch.randn((1, 512), dtype=torch.float, device=device)
            id_emb = F.normalize(id_emb, p=2, dim=1)
        else:
            id_emb = self.get_fae_id(source)[0:1]

        vr = VideoDecoder(vfp, device="cuda")
        num_frames = vr.metadata.num_frames_from_content
        device = self.align_dst.device

        pbar = tqdm(range(vr.metadata.num_frames_from_content), unit="frame")

        slices = [(i, min(i + batch_size, num_frames)) for i in range(0, num_frames, batch_size)]

        try:
            for slic in slices:
                pbar.n = slic[1]
                pbar.refresh()

                chunk = vr.get_frames_in_range(*slic).data.to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0)

                aligned, norm_affine_matrix, idxs = self.aligner.align(list(chunk.unbind(dim=0)), (self.size,) * 2, self.align_dst)

                swaped_face, face_mask = self.swap_single(aligned, id_emb[:1])
                fake_id = self.idencoder.extract_id_feats(swaped_face, face_mask)
                similarity = F.cosine_similarity(fake_id, id_emb.expand(fake_id.shape[0], -1))

                pbar.set_description_str(f"相似度: {similarity.mean():.4f}")
                q.put(swaped_face)

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            exit()

        q.put(None)


if __name__ == "__main__":
    from loguru import logger

    from misc.utils import DataFolder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device=}")
    dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024")
    # dataset = DataFolder("IDAssets")
    logger.info(f"{dataset=}")

    ckpt = "log/train_7_retrain/ckpt/1260000.pth"
    logger.info(f"{ckpt=}")

    target = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-02-17(袁忠 曹属馨)/20230217-160724910曹属馨(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-06-20(邓叮玲 许馨文)/20230620-155807793邓叮玲(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-08-28(李小姐 邱小姐)/20230828-122302李小姐(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2025-07-03(蒋女士 林嘉惠 王钰 符媛媛 石星儿)/王钰(电脑录屏).mp4"
    # target = "IDAssets/抖音-记录美好生活.mp4"

    source = dataset.sample(rando=True)
    source = "IDAssets/刘亦菲.png"
    logger.info(f"{source=}")

    swap = Swapper(ckpt).to(device=device)
    # swap.swap_folder("IDAssets", source, sf_prefix="train_7_retrain")
    swap.swap_video(target, source)
