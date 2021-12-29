import os
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image

#imagenet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=256//20*2+1, sigma=(0.1, 2.0)),
    transforms.Normalize([*mean], [*std])
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([*mean], [*std])
])


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

class LabelDataset(Dataset):
    def __init__(self, annotation_filepath, img_dir, transform=None):
        self.path = img_dir
        self.label = pd.read_csv(annotation_filepath)
        self.transform = transform
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        img1_name, img2_name = f'{self.label.loc[idx][0][:12]}.npy', f'{self.label.loc[idx][0][13:]}.npy'
        label = self.label.loc[idx][1]
        img1 = np.load(os.path.join(self.path, img1_name))
        img2 = np.load(os.path.join(self.path, img2_name))
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, label

class TestDataset(Dataset):
    def __init__(self, query_filepath, img_dir, transform=valid_transform):
        self.path = img_dir
        self.query = pd.read_csv(query_filepath, header=None)
        self.transform = transform
    def __len__(self):
        return len(self.query)
    def __getitem__(self, idx):
        img1_name, img2_name = self.query.loc[idx]
        query = f'{img1_name[:-4]}_{img2_name[:-4]}'
        img1 = np.load(os.path.join(self.path, img1_name[:-4]+'.npy'))
        img2 = np.load(os.path.join(self.path, img2_name[:-4]+'.npy'))
        img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, query

