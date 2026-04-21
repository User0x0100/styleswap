from typing import Any
from torchcodec.decoders import VideoDecoder

import multiprocessing as mp
from pathlib import Path
from torchvision.io import decode_image
import cv2
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import autocast, nn
from tqdm import tqdm

from dfm.model.saehd.arch import DFUDT, LIAEUDT
from misc.tools import FaceAlign
from misc.model import IDEncoder
from .networks.generator import Generator
from misc.model import FaceParsing


class Swapper(nn.Module):

    def __init__(self, ckpt: str, device: str = "cuda:0"):
        super().__init__()

        self.device = torch.device(device)

        print(f"Loading checkpoint from {ckpt}")
        ckpt: dict[str, Any] = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)

        self.iter, self.size = (ckpt[k] for k in ("iter", "size"))

        print(f"Weight Info:\n" f"  {'iter':25}: {self.iter}\n" f"  {'size':25}: {self.size}\n")

        self.net_g = Generator(input_res=self.size).to(self.device)
        self.net_g.load_state_dict(ckpt["net_g"])

        self.aligner = FaceAlign(from_unit_range=True).to(self.device)

        self.idencoder = IDEncoder.Model(provider=IDEncoder.PROVIDER.ARC_OFFICIAL_R50).to(self.device)

        align_dst = IDEncoder.get_dst_pts(self.size)

        self.facepars = FaceParsing.Model(compile_model=False).to(self.device)

        self.register_buffer("align_dst", align_dst, persistent=False)

        self.eval()
        self.requires_grad_(False)

    def get_fae_id(self, fp: str) -> torch.Tensor:
        device = self.align_dst.device

        image = decode_image(fp)
        image = image.to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0).unsqueeze_(0)
        align_face, _, _ = self.aligner.align(images=image, out_size=(IDEncoder.SIZE,) * 2, dst_pts5_pixel=IDEncoder.get_dst_pts(IDEncoder.SIZE).to(device=device))

        id_emb = self.idencoder(align_face)
        if id_emb.size(0) > 1:
            id_emb = id_emb[:1]

        return id_emb

    @staticmethod
    def display_worker(q: mp.Queue, fps=25):

        delay = int(1000 / fps)

        while True:
            frames: Tensor = q.get()
            if frames is None:
                break

            # BGR
            frames = frames[:, [2, 1, 0], :, :]
            frames = frames.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()

            for frame in frames:
                cv2.imshow("swaped", frame)
                cv2.waitKey(delay)

    # @autocast(device_type="cuda")
    def swap_video(self, vfp: str, idfp: str, batch_size: int = 32):

        device = self.align_dst.device

        mp.set_start_method("spawn", force=True)
        q = mp.Queue(maxsize=15)
        p = mp.Process(target=self.display_worker, args=(q,), daemon=True)
        p.start()

        vr = VideoDecoder(vfp, device="cuda")
        num_frames = vr.metadata.num_frames_from_content
        device = self.align_dst.device

        id_emb = self.get_fae_id(idfp)

        pbar = tqdm(range(vr.metadata.num_frames_from_content), unit="frame")

        slices = [(i, min(i + batch_size, num_frames)) for i in range(0, num_frames, batch_size)]

        try:
            for slic in slices:
                pbar.n = slic[1]
                pbar.refresh()

                chunk = vr.get_frames_in_range(*slic).data.to(device=device, dtype=torch.float).div_(127.5).sub_(1.0)

                aligned, norm_affine_matrix, idxs = self.aligner.align(list(chunk.unbind(dim=0)), (self.size,) * 2, self.align_dst)

                # mask = self.facepars(aligned)

                B = aligned.size(0)
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    swaped: Tensor = self.net_g(aligned, id_emb.expand(B, -1))

                # inv_mask = 1.0 - mask

                # swaped *= mask

                # bg = aligned * inv_mask

                # swaped += bg

                concat = torch.cat((aligned, swaped), dim=3)
                q.put(concat)

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            exit()

        q.put(None)


if __name__ == "__main__":
    device = torch.device("cuda", 0)
    ckpt = "log_dfm/train_facedancer/ckpt/770000.pth"
    swap = Swapper(ckpt).to(device=device)
    dst = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    # dst = "/opt/share/deepfake/linshi/录屏素材/2023-02-17(袁忠 曹属馨)/20230217-160724910曹属馨(电脑录屏).mp4"
    # dst = "/opt/share/deepfake/linshi/录屏素材/2023-06-20(邓叮玲 许馨文)/20230620-155807793邓叮玲(电脑录屏).mp4"
    # dst = "/opt/share/deepfake/linshi/录屏素材/2023-08-28(李小姐 邱小姐)/20230828-122302李小姐(电脑录屏).mp4"
    # dst = "/opt/share/deepfake/linshi/录屏素材/2025-07-03(蒋女士 林嘉惠 王钰 符媛媛 石星儿)/王钰(电脑录屏).mp4"
    # dst = "/opt/share/deepfake/dataset_1/oneman/1.mp4"
    # swap.swap_video(dst, "faceset/ljx/arcface_pts/610_0.png")
    swap.swap_video(dst, "IDAssets/wuyanzu.png")
