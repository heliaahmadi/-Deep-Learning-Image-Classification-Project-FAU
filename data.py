from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from skimage.transform import resize

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, dataframe, mode, target_height=None, verbose=True):
        self.data = dataframe
        self.mode = mode
        self.verbose = verbose
        self.target_height = target_height  # If None, no resizing applied

        if self.mode == "train":
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomVerticalFlip(p=0.5),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])
        else:
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]
        image = imread(img_name)

        # Resize if target height is given
        if self.target_height is not None:
            original_height, original_width = image.shape[:2]
            aspect_ratio = original_width / original_height
            target_width = int(self.target_height * aspect_ratio)

            if self.verbose:
                print(f"[INFO] Resizing image {img_name}")
                print(f"[INFO] Original size: ({original_height}, {original_width})")
                print(f"[INFO] New size: ({self.target_height}, {target_width})")

            image = resize(image, (self.target_height, target_width), preserve_range=True, anti_aliasing=True)
            image = image.astype(np.uint8)

        image = gray2rgb(image)
        image = self.transform(image)

        label = np.zeros(2, dtype=int)
        label[0] = int(self.data.iloc[idx, 1:2].iloc[0])
        label[1] = int(self.data.iloc[idx, 2:].iloc[0])
        label = torch.from_numpy(label)

        return image, label
