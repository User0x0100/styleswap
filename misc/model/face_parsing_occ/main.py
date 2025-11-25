from pathlib import Path
import segmentation_models_pytorch as smp
import torch
from torch.nn import functional as F
from misc.utils import DataFolder
import torchvision.utils as utils

# https://github.com/face3d0725/FaceExtraction/blob/main/Dataset/utils.py
# "resnet18": {
#     "imagenet": {
#         "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
#         "input_space": "RGB",
#         "input_size": [3, 224, 224],
#         "input_range": [0, 1],
#         "mean": [0.485, 0.456, 0.406],
#         "std": [0.229, 0.224, 0.225],
#         "num_classes": 1000,
#     },


ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
CLASSES = 1
ATTENTION = None
ACTIVATION = None
DEVICE = "cuda:0"


if __name__ == "__main__":
    # model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, decoder_attention_type=ATTENTION, classes=CLASSES, activation=ACTIVATION)
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, decoder_attention_type=ATTENTION, classes=CLASSES, activation=ACTIVATION)

    encoder_state_dict = torch.load(Path(__file__).with_name("resnet18-5c106cde.pth"), weights_only=False)
    model.encoder.load_state_dict(encoder_state_dict)

    state_dict = torch.load(Path(__file__).with_name("epoch_16_best.ckpt"))
    state_dict = {cleaned: v for k, v in state_dict.items() if (cleaned := k.removeprefix("module."))}
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)

    dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024").iter_batch_with_tensor(batch_size=16, rando=True)
    image = torch.stack(next(dataset))
    image = image.to(device=DEVICE, dtype=torch.float).div_(255.0)
    input = F.interpolate(image, size=224, mode="bicubic")

    save_folder = Path(__name__).with_name("visi")
    save_folder.mkdir(parents=True, exist_ok=True)
    output = model(input)
    utils.save_image(output, save_folder / f"face_parseing_occ.png", nrow=16)
