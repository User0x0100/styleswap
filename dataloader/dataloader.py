import random
from nvidia.dali.types import DALIImageType, DALIDataType
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from typing import List, Tuple
from nvidia.dali.types import Constant
from nvidia.dali import pipeline_def
import numpy as np
import time
from misc.utils import DataFolder

from typing import List, Tuple, Union
import math


class ExternalInputIterator(object):
    def __init__(
        self,
        target_folders: Union[List[str], List[Tuple[str, float]]],
        source_folders: Union[List[str], List[Tuple[str, float]]],
        rando: bool = True,
    ):

        self.target_folders, self.target_weight = self.collect_folder(target_folders)
        self.source_folders, self.source_weight = self.collect_folder(source_folders)
        self.rando = rando
        self.print_folders_info()

    def __call__(self, sample_info):
        _ = sample_info
        target_folder = random.choices(self.target_folders, weights=self.target_weight, k=1)[0]
        source_folder = random.choices(self.source_folders, weights=self.source_weight, k=1)[0]

        target_file_path = target_folder.sample(self.rando)
        source_file_path = source_folder.sample(self.rando)

        target = np.fromfile(target_file_path, dtype=np.uint8)
        source = np.fromfile(source_file_path, dtype=np.uint8)

        return target, source

    @staticmethod
    def collect_folder(folder_weight_list: Union[List[str], List[Tuple[str, float]]]):
        readers: List[DataFolder] = []
        file_counts: List[int] = []
        adjustments: List[float] = []

        # 判断输入类型
        if isinstance(folder_weight_list[0], str):  # 纯文件夹路径
            for folder in folder_weight_list:
                reader = DataFolder(folder)
                readers.append(reader)
                file_counts.append(len(reader))
                adjustments.append(0.0)  # 默认无调整
        else:  # 带权重
            for folder, adj in folder_weight_list:
                reader = DataFolder(folder)
                readers.append(reader)
                file_counts.append(len(reader))
                adjustments.append(adj)  # [-1.0, 1.0]

        total_files = sum(file_counts)
        if total_files == 0:
            raise ValueError("所有文件夹为空，无法分配权重")

        # ✅ 基于文件数量的开方缩放计算基础权重
        sqrt_counts = [count**0.5 for count in file_counts]
        sqrt_sum = sum(sqrt_counts)
        base_weights = [sc / sqrt_sum for sc in sqrt_counts]

        # ✅ 权重调整：指数映射
        # adj = -1 → factor = 0.5, adj = 0 → 1, adj = 1 → 2
        adjusted_weights = [w * (2**adj) for w, adj in zip(base_weights, adjustments)]

        # ✅ 归一化
        weight_sum = sum(adjusted_weights)
        if weight_sum == 0:
            weights = [1.0 / len(adjusted_weights)] * len(adjusted_weights)
        else:
            weights = [w / weight_sum for w in adjusted_weights]

        return readers, weights

    def print_folders_info(self):
        from tabulate import tabulate

        def build_table(title, folders, weights):
            table_data = [[folder.folder, folder.len, f"{weight:.4f}"] for folder, weight in zip(folders, weights)]
            print(f"\n{title}")
            print(tabulate(table_data, headers=["Path", "Count", "Weight"], tablefmt="fancy_grid"))

        build_table("Target Folders", self.target_folders, self.target_weight)
        build_table("Source Folders", self.source_folders, self.source_weight)


@pipeline_def(enable_conditionals=True)
def swap_pipe(
    resize: int,
    target_folders: List[Tuple[str, float]],
    source_folders: List[Tuple[str, float]],
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    flip_prob: float = 0.5,
    same_image_prob: float = 0.1,
    rando: bool = True,
):

    external_source = fn.external_source(
        source=ExternalInputIterator(target_folders, source_folders, rando),
        num_outputs=2,
        device="cpu",
        no_copy=True,
        parallel=True,
        prefetch_queue_depth=32,
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

        # src = fn.color_twist(
        #     src,
        #     device="gpu",
        #     brightness=fn.random.uniform(range=[1.0 - 0.2, 1.0 + 0.2]),
        #     contrast=fn.random.uniform(range=[1.0 - contrast, 1.0 + contrast]),
        #     saturation=fn.random.uniform(range=[1.0 - saturation, 1.0 + saturation]),
        # )

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
            ("/opt/share/deepfake/wang_train_data2/大侧脸/刘亦菲侧脸2", 0.0),
            ("/opt/share/deepfake/wang_train_data2/大侧脸/renwu_dace", 0.0),
            ("/opt/share/deepfake/stars/bcjz/aligned-clean-all", 0.0),
            ("/opt/share/deepfake/stars/hsx", 0.0),
            ("/opt/share/deepfake/lsy-close-eyes", 0.0),
            ("/opt/share/deepfake/all_dataSet/男女混合/aligned", 0.0),
            ("/opt/share/deepfake/all_dataSet/baidu爬取男性45度侧脸及各种风格脸型dst(已经切脸，没有mask)", 0.0),
            ("/opt/share/deepfake/dataset_1/ffhq_1024", 0.0),
            ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024", 0.0),
            ("/opt/share/deepfake/dataset_1/vggface2_hq512", 0.0),
            # ("/opt/share/deepfake/dataset_1/closed_eyes", 0.25),
            # ("/opt/share/deepfake/dataset_1/large_angle", 0.25),
        ],
        source_folders=[
            ("/opt/share/deepfake/dataset_1/ffhq_1024", 0.0),
            ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024", 0.0),
            ("/opt/share/deepfake/dataset_1/vggface2_hq512", 0.0),
            ("/opt/share/deepfake/all_dataSet/男女混合/aligned", 0.0),
            ("/opt/share/deepfake/all_dataSet/男", 0.0),
            ("/opt/share/deepfake/all_dataSet/女/aligned", 0.0),
            ("/opt/share/deepfake/all_dataSet/黄种人-FFHQ", 0.0),
            ("/opt/share/deepfake/all_dataSet/generated_yellow-stylegan2", 0.0),
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
