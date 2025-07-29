import torch
import torch.nn.functional as F
from torch import nn

from misc.model import PTS682D, ArcFace
from misc.tools import compute_similarity_transform, pixel_to_normalized


class IDLoss(nn.Module):
    def __init__(self, weight: float = 1.0, realign: bool = True):
        super().__init__()
        self.weight = weight
        self.realign = realign
        self.arcface = ArcFace.Model()
        self.pts682d = PTS682D.Model()

        self.arcface.eval()
        self.pts682d.eval()
        self.arcface.requires_grad_(False)
        self.pts682d.requires_grad_(False)

        dst_pts_norm = pixel_to_normalized(ArcFace.ARCFACE_DST_PTS, [112, 112])

        self.register_buffer("dst_pts_norm", dst_pts_norm, persistent=False)

    def extract_id_feats(self, face: torch.Tensor, face_mask: torch.Tensor | None = None):
        if self.realign:
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
                face = face * face_mask

        return self.arcface(face)

    def forward(self, fake_id, real_id):
        loss = (1 - F.cosine_similarity(fake_id, real_id)).mean()
        return loss * self.weight


if __name__ == "__main__":

    import torch
    import torch.nn.functional as F

    from misc.model import FaceParsing
    from misc.utils import DataFolder, Timer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024")

    id_loss = IDLoss().to(device=device)
    facepars = FaceParsing.Model().to(device=device)

    data, filename = next(dataset.batch_iterator(batch_size))

    data = torch.stack(data)

    data = data.to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0)

    mask = facepars.process(data)

    id_loss.extract_id_feats(data, mask)
