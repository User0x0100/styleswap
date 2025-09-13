import torch
import torch.nn.functional as F
from torch import nn

from misc.model import PTS682D, IDEncoder
from misc.model.idencoder.main import PROVIDER
from misc.tools import compute_similarity_transform, pixel_to_normalized


class IDLoss(nn.Module):
    def __init__(self, weight: float = 1.0, id_encode_provider: PROVIDER = PROVIDER.ARC_OFFICIAL_R50):
        super().__init__()
        self.weight = weight

        self.arcface = IDEncoder.Model(provider=id_encode_provider)
        self.pts682d = PTS682D.Model()

        dst_pts_norm = pixel_to_normalized(IDEncoder.ARCFACE_DST_PTS, [112, 112])

        self.register_buffer("dst_pts_norm", dst_pts_norm, persistent=False)

        self.eval()
        self.requires_grad_(False)

    def extract_id_feats(self, face: torch.Tensor, face_mask: torch.Tensor | None = None, realign: bool = True):
        if realign:
            with torch.no_grad():
                B, _, H, W = face.shape
                pts68 = self.pts682d(face)

                src = PTS682D.convert_68_to_5(pts68)
                src = pixel_to_normalized(src, [H, W])
                dst = self.dst_pts_norm.unsqueeze(0).expand(B, -1, -1)

                M = compute_similarity_transform(dst, src)

                face_grid = F.affine_grid(M, face.size(), align_corners=False)

                if face_mask is not None:
                    mask_grid = F.affine_grid(M, face_mask.size(), align_corners=False)
                    face_mask = F.grid_sample(face_mask, mask_grid, align_corners=False, mode="bicubic")

            face = F.grid_sample(face, face_grid, align_corners=False, mode="bicubic")

        if face_mask is not None:
            face = face * face_mask + (1.0 - face_mask) * (-1.0)

        return self.arcface(face)

    def forward(self, fake_id, real_id):
        loss = (1 - F.cosine_similarity(fake_id, real_id)).mean()
        return loss * self.weight


if __name__ == "__main__":

    import torch
    import torch.nn.functional as F

    from misc.model import FaceParsing
    from misc.swapface import Swapper
    from misc.utils import DataFolder, Timer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 26
    dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024")

    id_loss_sim = IDLoss(id_encode_provider=PROVIDER.ARC_SIMSWAP).to(device=device)
    id_loss_off = IDLoss(id_encode_provider=PROVIDER.ARC_OFFICIAL_R50).to(device=device)
    id_loss_bf = IDLoss(id_encode_provider=PROVIDER.BLENDFACE).to(device=device)

    facepars = FaceParsing.Model(compile_model=False).to(device=device)
    swapper = Swapper("log/train_6/ckpt/260000.pth").to(device=device)

    a = dataset.sample_tensor(rando=False, label=False).unsqueeze_(0).to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0)
    b = dataset.sample_tensor(rando=False, label=False).unsqueeze_(0).to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0)

    a = F.interpolate(a, size=512, mode="bicubic", align_corners=False)
    b = F.interpolate(b, size=512, mode="bicubic", align_corners=False)

    mask_a = facepars.process(a)
    mask_b = facepars.process(b)

    id_emb = id_loss_off.extract_id_feats(a, mask_a, realign=True)

    fake = swapper.swap_single(b, id_emb)

    id_a = id_loss_sim.extract_id_feats(a, mask_a, realign=True)
    id_fake = id_loss_sim.extract_id_feats(fake, mask_b, realign=True)

    sim = F.cosine_similarity(id_a, id_fake, dim=1)
    print(f"simswap: {sim.item()}")

    id_a = id_loss_bf.extract_id_feats(a, realign=True)
    id_fake = id_loss_bf.extract_id_feats(fake, realign=True)

    sim = F.cosine_similarity(id_a, id_fake, dim=1)
    print(f"blendface: {sim.item()}")
