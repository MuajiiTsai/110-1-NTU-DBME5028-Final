import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import argparse
import tools
from models import SimTriplet
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from dataset import *
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--img_dir', default='./train_npy', type=str, help='image directory')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--save_model', default='./output', type=str)

args = parser.parse_args()
img_dir = args.img_dir
epochs, batch_size = args.epoch, args.batch_size
model_path = args.save_model

#prepare data
train_dataset = datasettriplet(img_dir=img_dir, transform=train_transform)
valid_dataset = datasettriplet(img_dir=img_dir, transform=valid_transform)

split = int(len(train_dataset)*0.2)
indices = [i for i in range(len(train_dataset))]
train_idx, valid_idx = indices[split:], indices[:split]

train_indices, valid_indices = train_test_split(indices)
train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = SimTriplet()
# model = model.cuda()

learning_rate = 1e-3
best_valid_loss = np.inf

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

for num_epoch in range(epochs):
    model.train()
    train_losses = []
    for img1, img2, img3 in tqdm(train_loader):
        optimizer.zero_grad()
        # img1, img2, img3 = img1.cuda(), img2.cuda(), img3.cuda()
        loss_dict = model(img1, img2, img3)
        loss = loss_dict['loss'].mean()
        loss.backward()
        optimizer.step()
        # loss = loss.detach().cpu().numpy()
        train_losses.append(loss)
    train_loss = np.mean(train_losses)
    print(f"Training loss: {train_loss}    {num_epoch}/{epochs}")

    model.eval()
    valid_losses = []
    for img1, img2, img3 in tqdm(valid_loader):
        # img1, img2, img3 = img1.cuda(), img2.cuda(), img3.cuda()
        loss_dict = model(img1, img2, img3)
        loss = loss_dict['loss'].mean()
        # loss = loss.detach().cpu().numpy()
        valid_losses.append(loss)
    valid_loss = np.mean(valid_losses)
    print(f"Valid loss: {valid_loss}    {num_epoch}/{epochs}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_path)
        print("The model is saved")