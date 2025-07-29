import os
import random
import time
from functools import wraps
from pathlib import Path
from typing import List, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.io import decode_image
from torchvision.io.image import ImageReadMode


class Timer:
    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        if self.verbose:
            label = self.name or "Block"
            print(f"[{label}] Elapsed: {self.format_time(self.elapsed)}")

    def __call__(self, func):
        """使 Timer 实例可作为装饰器使用"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            label = self.name or func.__name__
            with Timer(name=label, verbose=self.verbose):
                return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds >= 1:
            return f"{seconds:.3f} s"
        elif seconds >= 1e-3:
            return f"{seconds * 1e3:.3f} ms"
        else:
            return f"{seconds * 1e6:.3f} µs"


import re


def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", filename)]


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
        self.files = sorted(self.files, key=natural_sort_key)
        self.len = len(self.files)
        self.idx = 0

        if self.len == 0:
            raise FileNotFoundError(f"No valid image files found in folder: {folder}")

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self) -> Path:
        if self.idx >= self.len:
            raise StopIteration
        fp = Path(self.folder) / self.files[self.idx]
        self.idx += 1
        return fp

    def __getitem__(self, index: int) -> Path:
        if 0 <= index < self.len:
            fp = Path(self.folder) / self.files[index]
            return fp
        else:
            raise IndexError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"DataFolder(path='{self.folder}', samples={self.len})"

    def sample(self, rando: bool = False) -> Path:

        if rando:
            fp = self[random.randint(0, self.len - 1)]
        else:
            if self.idx >= self.len:
                self.idx = 0
            fp = next(self)

        return fp

    def sample_tensor(self, rando: bool = False, label: bool = False) -> torch.Tensor | tuple[torch.Tensor, str]:

        fp = self.sample(rando)

        if label:
            return decode_image(fp, ImageReadMode.RGB), fp.name
        else:
            return decode_image(fp, ImageReadMode.RGB)

    def iter_batch_with_tensor(self, batch_size: int, label: bool = False, rando: bool = False):
        indices = list(range(self.len))
        if rando:
            random.shuffle(indices)

        for start in range(0, self.len, batch_size):
            images: List[torch.Tensor] = []
            labels: List[str] = []
            for i in indices[start : start + batch_size]:
                fp = Path(self.folder) / self.files[i]
                img = decode_image(fp, mode=ImageReadMode.RGB)
                images.append(img)
                if label:
                    labels.append(fp.name)
            yield (images, labels) if label else images


if __name__ == "__main__":
    data = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024")
    for d, n in data.batch_iterator(3):
        print(n)
        exit(0)
