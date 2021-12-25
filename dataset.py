import os
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image

#imagenet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class datasetpair(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = os.listdir(img_dir)
        self.transform = transform
        self.path = img_dir
    def __len__(self):
        return len(self.img_dir)
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.img_dir[idx])
        img = np.load(img_path)
        img = Image.fromarray(img)
        img1, img2 = self.transform(img), self.transform(img)
        return img1, img2

class datasettriplet(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = os.listdir(img_dir)
        self.transform = transform
        self.path = img_dir
    def __len__(self):
        return len(self.img_dir)
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.img_dir[idx])
        img = np.load(img_path)
        img = Image.fromarray(img)
        img1, img2 ,img3 = self.transform(img), self.transform(img), self.transform(img)
        return img1, img2, img3


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    # transforms.Normalize([*mean], [*std])
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([*mean], [*std])
])

