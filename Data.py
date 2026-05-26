import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class RealRainDataset(Dataset):
    def __init__(self, root_dir, input_dir='input', target_dir='target', patch_size=128, split='train'):
        super().__init__()
        self.root_dir = root_dir
        self.input_dir = os.path.join(self.root_dir, input_dir)
        self.target_dir = os.path.join(self.root_dir, target_dir)
        self.patch_size = patch_size
        self.split = split
        self.samples = self._load_samples()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def _load_samples(self):
        input_imgs = sorted([os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        target_imgs = sorted([os.path.join(self.target_dir, f) for f in os.listdir(self.target_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        samples = []
        for inp, tar in zip(input_imgs, target_imgs):
            inp_name = os.path.splitext(os.path.basename(inp))[0]
            tar_name = os.path.splitext(os.path.basename(tar))[0]
            if inp_name == tar_name:
                samples.append({'rainy': inp, 'clear': tar})
            else:
                print(f"文件名不匹配: {inp} 与 {tar}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rainy_img = Image.open(sample['rainy']).convert("RGB")
        clear_img = Image.open(sample['clear']).convert("RGB")

        rainy = TF.to_tensor(rainy_img)
        clear = TF.to_tensor(clear_img)

        if self.split == 'train':
            # 填充处理
            _, h, w = rainy.shape
            ps = self.patch_size
            pad_h = (ps - h) if h < ps else 0
            pad_w = (ps - w) if w < ps else 0
            if pad_h > 0 or pad_w > 0:
                rainy = TF.pad(rainy, (0, 0, pad_w, pad_h), padding_mode='reflect')
                clear = TF.pad(clear, (0, 0, pad_w, pad_h), padding_mode='reflect')
                _, h, w = rainy.shape

            # 随机裁剪
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
            rainy = rainy[:, top:top+ps, left:left+ps]
            clear = clear[:, top:top+ps, left:left+ps]

            # 数据增强
            aug = random.randint(0, 7)
            if aug == 1:
                rainy = rainy.flip(1)  # 垂直翻转
                clear = clear.flip(1)
            elif aug == 2:
                rainy = rainy.flip(2)  # 水平翻转
                clear = clear.flip(2)
            elif aug == 3:
                 rainy = torch.rot90(rainy, k=1, dims=(1, 2))
                 clear = torch.rot90(clear, k=1, dims=(1, 2))
            elif aug == 4:
                 rainy = torch.rot90(rainy, k=2, dims=(1, 2))
                 clear = torch.rot90(clear, k=2, dims=(1, 2))
            elif aug == 5:
                 rainy = torch.rot90(rainy, k=3, dims=(1, 2))
                 clear = torch.rot90(clear, k=3, dims=(1, 2))
            elif aug == 6:
                 rainy = rainy.flip(1)
                 clear = clear.flip(1)
                 rainy = torch.rot90(rainy, k=1, dims=(1, 2))
                 clear = torch.rot90(clear, k=1, dims=(1, 2))
            elif aug == 7:
                 rainy = rainy.flip(2)
                 clear = clear.flip(2)
                 rainy = torch.rot90(rainy, k=1, dims=(1, 2))
                 clear = torch.rot90(clear, k=1, dims=(1, 2))

        # 标准化
        rainy = self.normalize(rainy)
        clear = self.normalize(clear)

        return {'Rainy': rainy, 'Clear': clear}

class Rain100LDataset(Dataset):
    def __init__(self, root_dir, input_dir='input', target_dir='target', patch_size=128, split='train'):
        super().__init__()
        self.root_dir = root_dir
        self.input_dir = os.path.join(self.root_dir, input_dir)
        self.target_dir = os.path.join(self.root_dir, target_dir)
        self.patch_size = patch_size
        self.split = split
        self.samples = self._load_samples()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def _load_samples(self):
        input_imgs = sorted([os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        target_imgs = sorted([os.path.join(self.target_dir, f) for f in os.listdir(self.target_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        samples = []
        for inp in input_imgs:
            # 从输入文件名提取基础ID (如 "norain-103x2" -> "norain-103")
            inp_name = os.path.splitext(os.path.basename(inp))[0]
            base_name = inp_name.replace("x2", "")  # 移除所有x2标记

            # 在target目录中查找匹配文件 (如 "norain-103.png")
            matched_tars = [t for t in target_imgs
                            if os.path.splitext(os.path.basename(t))[0] == base_name]

            if matched_tars:
                samples.append({'rainy': inp, 'clear': matched_tars[0]})
            else:
                print(f"警告: 找不到匹配目标文件: {inp}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rainy_img = Image.open(sample['rainy']).convert("RGB")
        clear_img = Image.open(sample['clear']).convert("RGB")

        rainy = TF.to_tensor(rainy_img)
        clear = TF.to_tensor(clear_img)

        if self.split == 'train':
            # 填充处理
            _, h, w = rainy.shape
            ps = self.patch_size
            pad_h = (ps - h) if h < ps else 0
            pad_w = (ps - w) if w < ps else 0
            if pad_h > 0 or pad_w > 0:
                rainy = TF.pad(rainy, (0, 0, pad_w, pad_h), padding_mode='reflect')
                clear = TF.pad(clear, (0, 0, pad_w, pad_h), padding_mode='reflect')
                _, h, w = rainy.shape

            # 随机裁剪
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
            rainy = rainy[:, top:top+ps, left:left+ps]
            clear = clear[:, top:top+ps, left:left+ps]

            # 数据增强
            aug = random.randint(0, 7)
            if aug == 1:
                rainy = rainy.flip(1)  # 垂直翻转
                clear = clear.flip(1)
            elif aug == 2:
                rainy = rainy.flip(2)  # 水平翻转
                clear = clear.flip(2)
            elif aug == 3:
                 rainy = torch.rot90(rainy, k=1, dims=(1, 2))
                 clear = torch.rot90(clear, k=1, dims=(1, 2))
            elif aug == 4:
                 rainy = torch.rot90(rainy, k=2, dims=(1, 2))
                 clear = torch.rot90(clear, k=2, dims=(1, 2))
            elif aug == 5:
                 rainy = torch.rot90(rainy, k=3, dims=(1, 2))
                 clear = torch.rot90(clear, k=3, dims=(1, 2))
            elif aug == 6:
                 rainy = rainy.flip(1)
                 clear = clear.flip(1)
                 rainy = torch.rot90(rainy, k=1, dims=(1, 2))
                 clear = torch.rot90(clear, k=1, dims=(1, 2))
            elif aug == 7:
                 rainy = rainy.flip(2)
                 clear = clear.flip(2)
                 rainy = torch.rot90(rainy, k=1, dims=(1, 2))
                 clear = torch.rot90(clear, k=1, dims=(1, 2))

        # 标准化
        rainy = self.normalize(rainy)
        clear = self.normalize(clear)

        return {'Rainy': rainy, 'Clear': clear}