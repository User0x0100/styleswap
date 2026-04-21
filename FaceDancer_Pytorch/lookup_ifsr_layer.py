from typing import Dict
import torch
import csv
from torch import nn, Tensor, autocast
import torch.nn.functional as F
from collections import defaultdict
from torchvision.io import decode_image
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from misc.tools import FaceAlign
from misc.model import IDEncoder


class IFSRExplorer:
    def __init__(self, backbone: nn.Module):

        self.backbone = backbone
        self._feats: Dict[str, Tensor] = {}
        self._hooks = []

        self._register_hooks()

    def _register_hooks(self):
        for layer_name_, module_ in self.backbone.named_children():
            for layer_name, module in module_.named_children():

                def make_hook(layer_name):
                    def hook(module, inputs, output):
                        self._feats[layer_name] = output

                    return hook

                h = module.register_forward_hook(make_hook(f"{layer_name_}.{layer_name}"))
                self._hooks.append(h)

    def get_feats(self) -> dict[str, Tensor]:
        return self._feats

    def clear(self):
        self._feats.clear()


def cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    a = a.flatten(1)
    b = b.flatten(1)
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return F.cosine_similarity(a, b, dim=1)


@torch.no_grad()
def ifsr(target_face: str, video_fp: str, enable_bf16: bool = True, csv_suffix: str = ""):
    batch_size = 32
    device = torch.device("cuda:0")
    idencoder = IDEncoder.Model(provider=IDEncoder.PROVIDER.BLENDFACE).to(device)
    feats_exper = IFSRExplorer(idencoder.backbone)
    aligner = FaceAlign(from_unit_range=True).to(device)
    align_pts = IDEncoder.get_dst_pts().to(device)

    target_face = decode_image(target_face).to(device=device, dtype=torch.float).div_(127.5).sub_(1.0).unsqueeze_(0)
    align_target_face = aligner.align(images=target_face, out_size=(IDEncoder.SIZE,) * 2, dst_pts5_pixel=align_pts)[0]

    id_emb: Tensor = idencoder(align_target_face)
    assert id_emb.size(0) == 1, "target_face检测到多个面部"

    target_face_feats = feats_exper.get_feats()
    target_face_feats = {k: v.clone() for k, v in target_face_feats.items()}

    vr = VideoDecoder(video_fp, device="cuda")
    num_frames = vr.metadata.num_frames_from_content

    pbar = tqdm(range(vr.metadata.num_frames_from_content), unit="frame")

    slices = [(i, min(i + batch_size, num_frames)) for i in range(0, num_frames, batch_size)]

    layer_sims: dict[str, list[Tensor]] = defaultdict(list)
    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enable_bf16):
        for slic in slices:
            pbar.n = slic[1]
            pbar.refresh()

            chunk = vr.get_frames_in_range(*slic).data.to(device=device, dtype=torch.float).div_(127.5).sub_(1.0)

            aligned = aligner.align(list(chunk.unbind(dim=0)), (IDEncoder.SIZE,) * 2, align_pts)[0]

            if aligned.size(0) > batch_size:
                print("检测到多个面部，丢弃该批次")
                continue

            id_emb: Tensor = idencoder(aligned)
            video_feats = feats_exper.get_feats()

            for layer_name in target_face_feats.keys():
                target_feat = target_face_feats[layer_name]
                video_feat = video_feats[layer_name]

                target_feat_expanded = target_feat.expand(video_feat.size(0), *target_feat.shape[1:])

                sim = cosine_sim(target_feat_expanded, video_feat)

                layer_sims[layer_name].append(sim.cpu())

            feats_exper.clear()

    pbar.close()

    final_stats = {}
    for layer_name, stats in layer_sims.items():

        all_sims = torch.cat(stats, dim=0)

        final_stats[layer_name] = {
            "max": all_sims.max().item(),
            "min": all_sims.min().item(),
            "mean": all_sims.mean().item(),
            "std": all_sims.std().item(),
            "median": all_sims.median().item(),
        }

    csv_fp = f"ifsr_layer_stats_{csv_suffix}.csv"

    with open(csv_fp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "max", "min", "mean", "std", "median"])

        for layer_name, stats in final_stats.items():
            writer.writerow(
                [
                    layer_name,
                    f"{stats['max']:.6f}",
                    f"{stats['min']:.6f}",
                    f"{stats['mean']:.6f}",
                    f"{stats['std']:.6f}",
                    f"{stats['median']:.6f}",
                ]
            )


if __name__ == "__main__":
    ifsr("/opt/share/deepfake/dataset_1/oneman/1_align_results/0_0.png", "/opt/share/deepfake/dataset_1/oneman/1.mp4", csv_suffix="BLENDFACE")
