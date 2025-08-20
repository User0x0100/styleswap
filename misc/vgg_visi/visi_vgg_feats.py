from pathlib import Path

import torch
import torchvision.utils as utils
from torch import nn as nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19

from misc.utils import DataFolder

size = 1024
batch_size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024").iter_batch_with_tensor(batch_size=batch_size, rando=True)


image = torch.stack(next(dataset))

image = image.to(device=device, dtype=torch.float).div_(255.0)
input = F.interpolate(image, size=size, mode="bicubic")

vggnet = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device=device).eval()

feats = []
for module in vggnet.features:
    input = module(input)
    feats.append(input)

from losses.vgg import NAMES

save_folder = Path(__name__).with_name("vgg_visi_result")
save_folder.mkdir(parents=True, exist_ok=True)

for layer_n, feat in zip(NAMES["vgg19"], feats):
    feat_img = feat.abs().mean(dim=1, keepdim=True)
    img = F.interpolate(image, size=feat_img.shape[-2], mode="bicubic")
    feat_img = feat_img.expand(-1, 3, -1, -1)
    save = torch.cat((img, feat_img))
    utils.save_image(save, save_folder / f"{layer_n}.png", nrow=batch_size)
