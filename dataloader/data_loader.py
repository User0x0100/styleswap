import os
import json
import hashlib
import random
from typing import List, Union, Tuple, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from face_parsing.face_parsing import FaceParsing


class WeightedDirectorySelector:
    """带权重的目录选择器"""

    def __init__(self, dirs_config: List[Tuple[str, float]]):
        self.dirs = [d[0] for d in dirs_config]
        self.weights = [d[1] for d in dirs_config]

        # 归一化权重
        total_weight = sum(self.weights)
        if abs(total_weight - 1.0) > 1e-6:
            print(f"权重总和 {total_weight:.4f} 不等于1，自动归一化")
            self.weights = [w / total_weight for w in self.weights]

        # 构建累积概率数组用于快速采样
        self.cumulative_weights = []
        cumsum = 0
        for w in self.weights:
            cumsum += w
            self.cumulative_weights.append(cumsum)

    def select_directory(self) -> str:
        """根据权重随机选择一个目录"""
        rand_val = random.random()
        for i, cum_weight in enumerate(self.cumulative_weights):
            if rand_val <= cum_weight:
                return self.dirs[i]
        # 容错：如果由于浮点精度问题没有选中，返回最后一个
        return self.dirs[-1]


class FaceDataset(Dataset):
    # 支持的图像文件扩展名
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(
        self,
        size: int,
        image_dirs: Optional[
            Union[str, List[str], List[Tuple[str, float]]]
        ] = None,  # 改为可选项
        target_dirs: Optional[
            Union[str, List[str], List[Tuple[str, float]]]
        ] = None,  # target专用目录
        source_dirs: Optional[
            Union[str, List[str], List[Tuple[str, float]]]
        ] = None,  # source专用目录
        cache_file: str = "dataset_cache.json",  # 缓存文件路径
        validate_cache: bool = False,
        same_image_prob: float = 0.25,  # target与source为同一张图片的概率
        flip_prob: float = 0.5,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.04,
        device: str = "cuda:0",  # GPU设备
    ):
        self.size = size
        self.cache_file = cache_file
        self.validate_cache = validate_cache
        self.device = device
        self.flip_prob = flip_prob
        self.same_image_prob = same_image_prob

        # 初始化人脸解析器 默认输出尺寸: 512
        self.face_parsing = FaceParsing(device)

        # 解析目录配置
        self.image_dirs_config = (
            self._parse_dirs_config(image_dirs) if image_dirs else []
        )
        self.target_dirs_config = (
            self._parse_dirs_config(target_dirs) if target_dirs else None
        )
        self.source_dirs_config = (
            self._parse_dirs_config(source_dirs) if source_dirs else None
        )

        # 验证至少有一个目录配置
        if (
            not self.image_dirs_config
            and not self.target_dirs_config
            and not self.source_dirs_config
        ):
            raise ValueError(
                "必须至少指定 image_dirs、target_dirs 或 source_dirs 中的一个"
            )

        # 验证target和source目录配置
        target_config = self.target_dirs_config or self.image_dirs_config
        source_config = self.source_dirs_config or self.image_dirs_config

        if not target_config:
            raise ValueError("没有可用的target目录配置")
        if not source_config:
            raise ValueError("没有可用的source目录配置")

        # 创建目录选择器
        self.target_selector = WeightedDirectorySelector(target_config)
        self.source_selector = WeightedDirectorySelector(source_config)

        # 获取所有需要缓存的目录
        all_dirs_to_cache = set()
        if self.image_dirs_config:
            all_dirs_to_cache.update([d[0] for d in self.image_dirs_config])
        if self.target_dirs_config:
            all_dirs_to_cache.update([d[0] for d in self.target_dirs_config])
        if self.source_dirs_config:
            all_dirs_to_cache.update([d[0] for d in self.source_dirs_config])

        # 加载目录文件缓存 (目录路径 -> 文件名列表)
        self.dir_files_cache = self._load_or_scan_directories(list(all_dirs_to_cache))

        # 计算数据集长度（使用target目录中的总文件数）
        self.length = sum(
            len(self.dir_files_cache.get(dir_path, [])) for dir_path, _ in target_config
        )

        print(f"数据集初始化完成:")
        self._print_directory_info()

        # 源图像的随机变换
        self.source_rand_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
            ]
        )

        # 颜色抖动，仅用于目标图像
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

        # 张量化与归一化
        self.to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )
        self.to_tensor_mask = transforms.ToTensor()

    def _parse_dirs_config(
        self, dirs_input: Union[str, List[str], List[Tuple[str, float]]]
    ) -> List[Tuple[str, float]]:
        """解析目录配置，返回(路径, 概率)的列表"""
        if isinstance(dirs_input, str):
            return [(dirs_input, 1.0)]

        if not dirs_input:
            return []

        if isinstance(dirs_input[0], str):
            # 如果是字符串列表，均等概率
            equal_prob = 1.0 / len(dirs_input)
            return [(d, equal_prob) for d in dirs_input]
        else:
            # 如果是元组列表，直接返回
            return list(dirs_input)

    def _print_directory_info(self):
        """打印目录信息"""

        def print_dir_config(name, config):
            print(f"  {name}:")
            for dir_path, weight in config:
                file_count = len(self.dir_files_cache.get(dir_path, []))
                print(f"    {dir_path}: {file_count} 个文件, 权重 {weight:.3f}")

        if self.image_dirs_config:
            print_dir_config("公共目录", self.image_dirs_config)
        if self.target_dirs_config:
            print_dir_config("Target目录", self.target_dirs_config)
        if self.source_dirs_config:
            print_dir_config("Source目录", self.source_dirs_config)

        print(f"  数据集长度: {self.length}")

    def _is_image_file(self, filename: str) -> bool:
        """检查文件是否为支持的图像格式"""
        return os.path.splitext(filename.lower())[1] in self.SUPPORTED_EXTENSIONS

    def _scan_single_directory(self, img_dir: str) -> List[str]:
        """扫描单个目录中的图像文件，返回文件名列表（不包含完整路径）"""
        filenames = []
        if not os.path.exists(img_dir):
            print(f"警告: 目录不存在 - {img_dir}")
            return filenames

        print(f"扫描目录: {img_dir}")
        for filename in os.listdir(img_dir):
            if self._is_image_file(filename):
                filenames.append(filename)

        print(f"在 {img_dir} 中找到 {len(filenames)} 个图像文件")
        return filenames

    def _load_cache(self) -> dict:
        """加载缓存文件"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                print(f"成功加载缓存文件: {self.cache_file}")
                return cache_data
            except Exception as e:
                print(f"加载缓存文件失败: {e}")
        return {}

    def _save_cache(self, cache_data: dict):
        """保存缓存文件"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"缓存已保存到: {self.cache_file}")
        except Exception as e:
            print(f"保存缓存文件失败: {e}")

    def _validate_cached_files(
        self, dir_path: str, cached_filenames: List[str]
    ) -> List[str]:
        """验证缓存中的文件是否仍然存在"""
        print(f"验证目录 {dir_path} 的缓存文件存在性...")
        import time

        start_time = time.time()
        valid_filenames = []

        for filename in cached_filenames:
            full_path = os.path.join(dir_path, filename)
            if os.path.exists(full_path):
                valid_filenames.append(filename)

        validation_time = time.time() - start_time
        print(f"验证完成，耗时 {validation_time:.2f} 秒")

        return valid_filenames

    def _load_or_scan_directories(self, all_dirs: List[str]) -> dict:
        """加载或扫描目录，返回 目录路径 -> 文件名列表 的字典"""
        cache_data = self._load_cache()
        dir_files_cache = {}
        dirs_to_scan = []

        # 检查每个目录的缓存
        for dir_path in all_dirs:
            if dir_path in cache_data:
                cached_filenames = cache_data[dir_path]
                print(f"找到目录 {dir_path} 的缓存，共 {len(cached_filenames)} 个文件")

                # 验证缓存的文件是否仍然存在
                if self.validate_cache:
                    valid_filenames = self._validate_cached_files(
                        dir_path, cached_filenames
                    )

                    if len(valid_filenames) == len(cached_filenames):
                        print(f"目录 {dir_path} 所有缓存文件均存在")
                        dir_files_cache[dir_path] = cached_filenames
                    else:
                        missing_count = len(cached_filenames) - len(valid_filenames)
                        print(
                            f"目录 {dir_path} 缓存中有 {missing_count} 个文件不存在，需要重新扫描"
                        )
                        dirs_to_scan.append(dir_path)
                else:
                    print("跳过目录缓存验证")
                    dir_files_cache[dir_path] = cached_filenames
            else:
                print(f"目录 {dir_path} 无缓存，需要扫描")
                dirs_to_scan.append(dir_path)

        # 扫描需要更新的目录
        cache_updated = False
        for dir_path in dirs_to_scan:
            filenames = self._scan_single_directory(dir_path)
            dir_files_cache[dir_path] = filenames

            # 更新该目录的缓存
            cache_data[dir_path] = filenames
            cache_updated = True

        # 保存更新的缓存
        if cache_updated:
            self._save_cache(cache_data)

        total_files = sum(len(filenames) for filenames in dir_files_cache.values())
        print(f"文件列表加载完成，共 {total_files} 个文件")
        return dir_files_cache

    def _get_random_file_from_dirs(self, selector: WeightedDirectorySelector) -> str:
        """从目录选择器中按权重选择目录，然后随机选择文件"""
        max_attempts = 10  # 最大尝试次数，防止无限循环

        for _ in range(max_attempts):
            # 按概率选择目录
            selected_dir = selector.select_directory()

            # 获取该目录下的文件列表
            filenames = self.dir_files_cache.get(selected_dir, [])

            if filenames:
                # 随机选择文件名
                selected_filename = random.choice(filenames)
                # 拼接完整路径
                return os.path.join(selected_dir, selected_filename)

        # 如果多次尝试都失败，随机选择一个非空目录
        non_empty_dirs = [d for d in selector.dirs if self.dir_files_cache.get(d, [])]
        if non_empty_dirs:
            fallback_dir = random.choice(non_empty_dirs)
            fallback_filename = random.choice(self.dir_files_cache[fallback_dir])
            print(f"警告: 使用备选目录 {fallback_dir}")
            return os.path.join(fallback_dir, fallback_filename)

        raise RuntimeError("无法找到任何有效的图像文件")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        _ = index
        # 1. 根据权重选择target图像
        target_path = self._get_random_file_from_dirs(self.target_selector)
        target = (
            Image.open(target_path)
            .resize([self.size, self.size], resample=Image.Resampling.LANCZOS)
            .convert("RGB")
        )

        # 2. 计算掩码
        target_mask = self.face_parsing.process(target).convert("L")

        # 3. 随机水平翻转
        if random.random() < self.flip_prob:
            target = F.hflip(target)
            target_mask = F.hflip(target_mask)

        # 4. 颜色抖动仅作用于目标图像
        target = self.color_jitter(target)

        # 5. 张量化与归一化
        target = self.to_tensor_norm(target)
        target_mask = self.to_tensor_mask(target_mask)

        # 6. 根据权重与概率选择source图像
        if random.random() < self.same_image_prob:
            is_same = True
            source = target

        else:
            is_same = False
            src_path = self._get_random_file_from_dirs(self.source_selector)
            source = (
                Image.open(src_path)
                .resize([self.size, self.size], resample=Image.Resampling.LANCZOS)
                .convert("RGB")
            )
            source = self.source_rand_transforms(source)
            source = self.to_tensor_norm(source)

        return target, source, target_mask, is_same

    def refresh_cache(self, specific_dirs: List[str] = None):
        """手动刷新缓存，重新扫描指定目录或所有目录"""
        all_dirs = set()
        if self.image_dirs_config:
            all_dirs.update([d[0] for d in self.image_dirs_config])
        if self.target_dirs_config:
            all_dirs.update([d[0] for d in self.target_dirs_config])
        if self.source_dirs_config:
            all_dirs.update([d[0] for d in self.source_dirs_config])

        dirs_to_refresh = specific_dirs if specific_dirs else list(all_dirs)
        print(f"手动刷新缓存: {dirs_to_refresh}")

        cache_data = self._load_cache()

        # 刷新指定目录
        for dir_path in dirs_to_refresh:
            if dir_path in all_dirs:
                filenames = self._scan_single_directory(dir_path)
                cache_data[dir_path] = filenames
                self.dir_files_cache[dir_path] = filenames
                print(f"目录 {dir_path} 缓存已更新，共 {len(filenames)} 个文件")

        # 重新计算数据集长度
        target_config = self.target_dirs_config or self.image_dirs_config
        self.length = sum(
            len(self.dir_files_cache.get(dir_path, [])) for dir_path, _ in target_config
        )

        # 保存缓存
        self._save_cache(cache_data)

        print(f"缓存刷新完成，数据集长度: {self.length}")

    def get_dataset_info(self):
        """获取数据集信息"""

        def get_dir_info(config):
            return [
                {
                    "path": dir_path,
                    "weight": weight,
                    "file_count": len(self.dir_files_cache.get(dir_path, [])),
                }
                for dir_path, weight in config
            ]

        info = {
            "total_length": self.length,
            "image_dirs": (
                get_dir_info(self.image_dirs_config) if self.image_dirs_config else None
            ),
            "target_dirs": (
                get_dir_info(self.target_dirs_config)
                if self.target_dirs_config
                else None
            ),
            "source_dirs": (
                get_dir_info(self.source_dirs_config)
                if self.source_dirs_config
                else None
            ),
        }
        return info


# 使用示例
if __name__ == "__main__":

    # 示例4: 原有的多目录配置
    dataset = FaceDataset(
        image_dirs=["/home/liaohaixun/styleswap_data/dateset/ffhq_1024"],
        # target_dirs=[
        #     ("filter_results/closed_eyes", 0.5),
        #     ("filter_results/large_angle", 0.5),
        # ],
        # source_dirs=[
        #     "/home/liaohaixun/styleswap_data/dateset/ffhq_1024",
        # ],
        size=512,
        cache_file="dataset_cache.json",
    )

    # 查看数据集信息
    print("数据集信息:")
    import json

    print(json.dumps(dataset.get_dataset_info(), indent=2, ensure_ascii=False))
