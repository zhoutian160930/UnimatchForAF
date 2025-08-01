import numpy as np
import random
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader

class WeakAugment1D:
    """一维序列的弱增强（如平移）"""
    def __init__(self, max_shift_ratio=0.3):
        self.max_shift_ratio = max_shift_ratio  # 平移比例范围（相对长度）

    def shift(self, signal):
        L = signal.shape[-1]
        max_shift = int(L * self.max_shift_ratio)
        shift_val = np.random.randint(-max_shift, max_shift + 1)
        if signal.ndim == 1:
            shifted = np.roll(signal, shift_val)
        else:
            shifted = np.roll(signal, shift_val, axis=-1)
        return shifted

    def __call__(self, signal):
        return self.shift(signal)


class StrongAugment1D:
    def __init__(self, noise_std=0.2, mask_ratio=0.1, seg_len_ratio=0.1):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.seg_len_ratio = seg_len_ratio

    def add_gaussian_noise(self, x):
        noise = np.random.normal(0, self.noise_std, x.shape)
        return x + noise

    def random_mask(self, x):
        x = x.copy()
        length = x.shape[1]
        mask_len = int(length * self.mask_ratio)
        start = np.random.randint(0, length - mask_len)
        x[:, start:start + mask_len] = 0
        return x

    def sigmoid_transform(self, x):
        x = x.copy()
        length = x.shape[1]
        seg_len = int(length * self.seg_len_ratio)
        start = np.random.randint(0, length - seg_len)
        x[:, start:start + seg_len] = 1 / (1 + np.exp(-x[:, start:start + seg_len]))
        return x

    def vertical_flip(self, x):
        return -x

    def horizontal_flip(self, x):
        return np.flip(x, axis=1)

    def __call__(self, x):
        # 随机选择 2 到 3 种增强方法
        all_methods = [0, 1, 2, 3, 4]
        selected = random.sample(all_methods, k=random.choice([2, 3]))

        x_aug = x.copy()
        for method_id in sorted(selected):
            if method_id == 0:
                x_aug = self.add_gaussian_noise(x_aug)
            elif method_id == 1:
                x_aug = self.random_mask(x_aug)
            elif method_id == 2:
                x_aug = self.sigmoid_transform(x_aug)
            elif method_id == 3:
                x_aug = self.vertical_flip(x_aug)
            elif method_id == 4:
                x_aug = self.horizontal_flip(x_aug)

        return x_aug
    
class NPYDataset(Dataset):
    def __init__(self, id_file, data_path, mode='labeled', transform=None):
        """
        Args:
            id_file: 包含每行一个 'index label' 的 txt 文件
            data_path: 统一的 .npy 文件路径，例如 'alldata.npy'
            mode: 'labeled', 'unlabeled' 或 'val'
            transform: 可选的变换函数
        """
        self.data = np.load(data_path, mmap_mode='r')  # 高效按需加载
        self.mode = mode
        self.transform = transform

        self.ids = []
        self.labels = []

        with open(id_file, 'r') as f:
            for line in f:
                idx, label = line.strip().split()
                self.ids.append(int(idx))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        data = self.data[self.ids[idx]]
        label = self.labels[idx]

        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)  # (1, L)

        if self.mode == 'unlabeled':
            weak_aug = WeakAugment1D()(data)
            strong_aug = StrongAugment1D()(data)

            weak_aug = torch.tensor(weak_aug.copy(), dtype=torch.float32)
            strong_aug = torch.tensor(strong_aug.copy(), dtype=torch.float32)

            return [weak_aug, strong_aug]

        else:
            if self.transform is not None:
                data = self.transform(data)

            data = torch.tensor(data.copy(), dtype=torch.float32)
            return data, label

