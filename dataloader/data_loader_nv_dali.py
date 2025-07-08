import os
import random
from nvidia.dali.types import DALIImageType, DALIDataType
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from typing import List, Tuple, Sequence
from nvidia.dali.types import Constant
from nvidia.dali import pipeline_def
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import cupy as cp


class DataFolder:
    def __init__(
        self,
        folder: str,
        file_filters: Sequence[str] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
        ),
    ):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if f.lower().endswith(file_filters)]
        self.len = len(self.files)

    def random_pick(self) -> str:
        idx = random.randint(0, self.len - 1)
        return os.path.join(self.folder, self.files[idx])


class ExternalInputIterator(object):
    def __init__(
        self,
        target_folders: List[Tuple[str, float]],
        source_folders: List[Tuple[str, float]],
    ):

        self.target_folders, self.target_weight = self.collect_folder(target_folders)
        self.source_folders, self.source_weight = self.collect_folder(source_folders)

    def __call__(self, sample_info):
        _ = sample_info
        target_folder = random.choices(self.target_folders, weights=self.target_weight, k=1)[0]
        source_folder = random.choices(self.source_folders, weights=self.source_weight, k=1)[0]

        target_file_path = target_folder.random_pick()
        source_file_path = source_folder.random_pick()

        target = np.fromfile(target_file_path, dtype=np.uint8)
        source = np.fromfile(source_file_path, dtype=np.uint8)

        return target, source

    @staticmethod
    def collect_folder(folder_weight_list: List[Tuple[str, float]]):

        readers: List[DataFolder] = []
        weights: List[float] = []
        for folder, weight in folder_weight_list:
            readers.append(DataFolder(folder))
            weights.append(weight)

        return readers, weights


@pipeline_def(enable_conditionals=True)
def swap_pipe(
    resize: int,
    target_folders: List[Tuple[str, float]],
    source_folders: List[Tuple[str, float]],
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    flip_prob: float = 0.5,
    same_image_prob: float = 0.1,  # target与source为同一张图片的概率
):

    external_source = fn.external_source(
        source=ExternalInputIterator(target_folders, source_folders),
        num_outputs=2,
        device="cpu",
        no_copy=True,
        parallel=True,
        prefetch_queue_depth=128,
        dtype=DALIDataType.UINT8,
        batch=False,
    )

    tgt_raw, src_raw = external_source[0:]

    tgt = fn.decoders.image(tgt_raw, device="mixed", output_type=DALIImageType.RGB, cache_type="largest", hw_decoder_load=0.75)
    tgt = fn.resize(tgt, device="gpu", size=resize, dtype=DALIDataType.FLOAT)

    if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
        tgt = fn.flip(tgt, device="gpu")

    tgt = fn.color_twist(
        tgt,
        device="gpu",
        brightness=fn.random.uniform(range=[1.0 - brightness, 1.0 + brightness]),
        contrast=fn.random.uniform(range=[1.0 - contrast, 1.0 + contrast]),
        saturation=fn.random.uniform(range=[1.0 - saturation, 1.0 + saturation]),
    )

    if fn.random.coin_flip(probability=same_image_prob, dtype=DALIDataType.BOOL):
        is_same = Constant(value=1.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])
        src = tgt
    else:
        is_same = Constant(value=0.0, device="gpu", dtype=DALIDataType.FLOAT, shape=[1])

        src = fn.decoders.image(src_raw, device="mixed", output_type=DALIImageType.RGB, cache_type="largest", hw_decoder_load=0.75)

        src = fn.resize(src, device="gpu", size=resize, dtype=DALIDataType.FLOAT)

        if fn.random.coin_flip(probability=flip_prob, dtype=DALIDataType.BOOL):
            src = fn.flip(src, device="gpu")

        src = fn.color_twist(
            src,
            device="gpu",
            brightness=fn.random.uniform(range=[1.0 - brightness, 1.0 + brightness]),
            contrast=fn.random.uniform(range=[1.0 - contrast, 1.0 + contrast]),
            saturation=fn.random.uniform(range=[1.0 - saturation, 1.0 + saturation]),
        )

    tgt = fn.normalize(tgt, device="gpu", mean=127.5, stddev=127.5)
    src = fn.normalize(src, device="gpu", mean=127.5, stddev=127.5)

    tgt = fn.transpose(tgt, device="gpu", perm=[2, 0, 1])
    src = fn.transpose(src, device="gpu", perm=[2, 0, 1])

    return tgt, src, is_same


if __name__ == "__main__":

    import time
    from torchvision import utils
    from nvidia.dali.plugin.pytorch import LastBatchPolicy
    import torch

    batch_size = 20

    pipe = swap_pipe(
        batch_size=batch_size,
        num_threads=4,
        prefetch_queue_depth=16,
        resize=512,
        device_id=0,
        target_folders=[
            # ("test_image_folder/closed_eyes", 1.0)
            ("/opt/share/deepfake/dataset_1/ffhq_1024", 0.333333),
            ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024", 0.333333),
            ("/opt/share/deepfake/dataset_1/vggface2_hq512", 0.333333),
            # ("/opt/share/deepfake/dataset_1/closed_eyes", 0.25),
            # ("/opt/share/deepfake/dataset_1/large_angle", 0.25),
        ],
        source_folders=[
            # ("test_image_folder/closed_eyes", 1.0)
            ("/opt/share/deepfake/dataset_1/ffhq_1024", 0.333333),
            ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024", 0.333333),
            ("/opt/share/deepfake/dataset_1/vggface2_hq512", 0.333333),
        ],
    )

    dali_iter = DALIGenericIterator(
        pipelines=pipe,
        output_map=["tgt", "src", "is_same"],
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.DROP,
    )

    for _ in range(5):
        dali_iter.next()

    time.sleep(1)

    print("执行性能测试...")
    times = []
    for i in range(5):
        start_time = time.time()
        dali_iter.next()
        times.append((time.time() - start_time) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    fps = 1000 / avg_time

    print(f"平均处理时间: {avg_time:.2f} ms")
    print(f"最快处理时间: {min_time:.2f} ms")
    print(f"处理帧率: {fps:.1f} FPS")

    (data,) = dali_iter.next()
    tgt, scr = data["tgt"], data["src"]
    print(tgt.shape)
    print(tgt.device)

    grid = torch.cat((tgt, scr), 0)
    utils.save_image(
        grid,
        "NVIDIA_DALI.png",
        nrow=batch_size,
        normalize=True,
        value_range=(-1, 1),
    )
