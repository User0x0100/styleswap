# ========================= ID LOSS =========================
# if enable_id_loss is not None:
#     from torchvision.io import decode_image
#     from torchvision.io.image import ImageReadMode
#     from misc.tools import FaceAlign
#     from misc.model import IDEncoder
#     from misc.model.idencoder.main import PROVIDER

#     if Path(enable_id_loss).exists() == False:
#         raise FileNotFoundError(f"ID Photo File: {enable_id_loss} Not found")

#     face = decode_image(enable_id_loss, ImageReadMode.RGB).to(device=self.device, dtype=torch.float).div_(255.0).unsqueeze(0)  # f32 1CHW [0.0 ~ 1.0]
#     align = FaceAlign(from_unit_range=True).to(self.device)
#     aligned_face, _, _ = align.align(face, (self.model.resolution,) * 2, IDEncoder.get_dst_pts(self.model.resolution).to(self.device))

#     self.id_loss = IDLoss(weight=0.001, id_encode_provider=PROVIDER.BLENDFACE).to(self.device)

#     with torch.no_grad():
#         tid = self.id_loss.get_id_feats(aligned_face, realign=False)[0:1].expand(batch_size, -1)
#         self.target_id = tid.detach().clone()


# if self.target_id is not None:
                #     pred_id = self.id_loss.get_id_feats(pred_src_dst, realign=False)
                #     src_dst_id_loss = self.id_loss(pred_id, self.target_id)
                #     self.log("src_dst_id_loss", src_dst_id_loss)
                #     G_loss += src_dst_id_loss