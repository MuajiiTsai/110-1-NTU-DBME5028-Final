import os
import numpy as np
import pandas as pd
import torch
import argparse
import tools
import random
import torchvision.models
import torch.nn.functional as F 
from models import SimTriplet
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Train SimTriplet')
parser.add_argument('--train_dir', default='./train_npy/', type=str, help='train data directory')
parser.add_argument('--test_dir', default='./test_npy/', type=str, help='test data directory')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epoch', default=1, type=int)

args = parser.parse_args()
train_dir, test_dir = args.train_dir,  args.test_dir
epochs, batch_size = args.epoch, args.batch_size

tools.same_seeds(456101)

output_filename = f'SimTriplet_E{epochs}_B{batch_size}'  

#prepare data
train_dataset = datasettriplet(img_dir=train_dir, transform=train_transform)
valid_dataset = datasettriplet(img_dir=train_dir, transform=valid_transform)

split = int(len(train_dataset)*0.2)
indices = [i for i in range(len(train_dataset))]
train_idx, valid_idx = indices[split:], indices[:split]

train_indices, valid_indices = train_test_split(indices)
train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

model = SimTriplet()
model = model.cuda()

learning_rate = 1e-3
best_valid_loss = np.inf

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

model_path = f'./output/{output_filename}.pth'
for num_epoch in range(epochs):
    model.train()
    train_losses = []
    for img1, img2, img3 in tqdm(train_loader):
        optimizer.zero_grad()
        img1, img2, img3 = img1.cuda(), img2.cuda(), img3.cuda()
        loss_dict = model(img1, img2, img3)
        loss = loss_dict['loss'].mean()
        loss.backward()
        optimizer.step()
        loss = loss.detach().cpu().numpy()
        train_losses.append(loss)
        del img1, img2, img3
    train_loss = np.mean(train_losses)
    print(f"Training loss: {train_loss}    {num_epoch+1}/{epochs}")

    model.eval()
    valid_losses = []
    for img1, img2, img3 in tqdm(valid_loader):
        img1, img2, img3 = img1.cuda(), img2.cuda(), img3.cuda()
        loss_dict = model(img1, img2, img3)
        loss = loss_dict['loss'].mean()
        loss = loss.detach().cpu().numpy()
        valid_losses.append(loss)
        del img1, img2, img3
    valid_loss = np.mean(valid_losses)
    print(f"Valid loss: {valid_loss}    {num_epoch+1}/{epochs}")

    if valid_loss < best_valid_loss:
        if not os.path.exists('output'):
            os.mkdir('output')
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_path)
        print("The model is saved")
    elif valid_loss > best_valid_loss+0.05:
        print("Maybe Overfitting")
        break

"""Finetune"""
#prepare data
label_dataset = LabelDataset(annotation_filepath='validation_ground_truth.csv',img_dir=train_dir, transform=valid_transform)
label_loader = DataLoader(label_dataset, batch_size=1, shuffle=False)

"""CHECK THIS"""

model = torchvision.models.resnet34()
save_dict = torch.load(model_path, map_location='cpu')
msg = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},strict=True)
print(msg)
model = model.cuda()

best_thrd = 0
for i in range(50):
    correct = 0
    thrd = random.uniform(0,1)
    for img1, img2, label in tqdm(label_loader):
        with torch.no_grad():
            img1, img2 = img1.cuda(), img2.cuda()
            feature1 = model(img1)
            feature2 = model(img2)
            cos_sim = F.cosine_similarity(feature1, feature2)
            cos_sim = cos_sim.detach().cpu().numpy()
            if(cos_sim > thrd):
                pred = 1
            else:
                pred = 0
            if pred == label:
                correct+=1
            del img1, img2, feature1, feature2
    grade = correct/len(label_loader)
    print(f"threshold: {thrd}")
    print(f"c-index: {grade}")
    if grade > best_grade:
        best_grade = grade
        best_thres = thrd
        print("The threshold is update.")
print("-------------------Terminal------------------------")
print(f"threshold: {best_thres}, c-index: {best_grade}")

#prediction

from torchvision.io import read_image, ImageReadMode
query_path = f'./queries.csv'
output_csv = f'./output/{output_filename}.csv'


model.eval()
test_dataset = TestDataset(query_path, test_dir)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle = False
)

output = pd.DataFrame(columns= ['query', 'prediction'])
with torch.no_grad():
    for img1, img2, query in tqdm(test_dataloader):
        img1, img2 = img1.cuda(), img2.cuda()
        feature1 = model(img1)
        feature2 = model(img2)
        cos_sim = F.cosine_similarity(feature1, feature2)
        cos_sim = cos_sim.detach().cpu().numpy()
        if(cos_sim > best_thres):
            temp = pd.DataFrame({
                'query': query,
                'prediction': 1
            }, index=[0])
        else:
            temp = pd.DataFrame({
                'query': query,
                'prediction': 0
            }, index=[0])
        output = output.append(temp, ignore_index=True)

output.to_csv(output_csv, index=False)
