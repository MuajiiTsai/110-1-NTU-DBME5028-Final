import os
import numpy as np
import pandas as pd
import torch
import argparse
import tools
import random
import torch.nn.functional as F 
from models import SimTriplet
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import *
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Train SimTriplet')
parser.add_argument('--train_dir', default='./train_npy/', type=str, help='train data directory')
parser.add_argument('--test_dir', default='./test_npy/', type=str, help='test data directory')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--filename', default='', type=str, help='output_filename: SimTriplet_E{args.epoch}_B{args.batch_size}_filename)')

args = parser.parse_args()
train_dir, test_dir = args.train_dir,  args.test_dir
epochs, batch_size = args.epoch, args.batch_size
filename = args.filename
tools.same_seeds(101)

output_filename = f'SimTriplet_E{epochs}_B{batch_size}_{filename}'  

#prepare data
train_dataset = datasettriplet(img_dir=train_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# create model

model_path = f'./output/{output_filename}.pth'
model = SimTriplet()
model = model.cuda()

learning_rate = 5e-3
best_train_loss = np.inf

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

def D(p, z): # negative cosine similarity
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

for num_epoch in range(epochs):
    print(f'epoch: {num_epoch+1}/{epochs}')
    model.train()
    train_losses = []
    for img1, img2, img3 in tqdm(train_loader):
        optimizer.zero_grad()
        img1, img2, img3 = img1.cuda(), img2.cuda(), img3.cuda()
        z1, p1 = model(img1)
        z2, p2 = model(img2)
        z3, p3 = model(img3)
        loss = D(p1, z2) / 2 + D(p2, z1) / 2 + D(p1, z3) / 2 + D(p3, z1) / 2
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        loss = loss.detach().cpu().numpy()
        train_losses.append(loss)
        del img1, img2, img3, z1, z2, z3, p1, p2, p3
    train_loss = np.mean(train_losses)
    print(f"Training loss: {train_loss}")
    
    if train_loss < best_train_loss:
        if not os.path.exists('output'):
            os.mkdir('output')
        best_train_loss = train_loss
        torch.save(model.state_dict(), model_path)
        print("The model is saved")
    elif train_loss > best_train_loss+0.05:
        print("Maybe Overfitting")
        break

"""Finetune"""
#prepare data
label_dataset = LabelDataset(annotation_filepath='validation_ground_truth.csv',img_dir=train_dir, transform=valid_transform)
label_loader = DataLoader(label_dataset, batch_size=1, shuffle=False,drop_last=True)


model = SimTriplet()
save_dict = torch.load(model_path)
model.load_state_dict(save_dict)
model = model.cuda()

model.eval()
best_thrd = 0
best_grade = 0
for i in range(50):
    correct = 0
    print(f'{i+1}/50: ')
    thrd = random.uniform(0,1)
    with torch.no_grad():
    	for img1, img2, label in tqdm(label_loader):
            img1, img2 = img1.cuda(), img2.cuda()
            feature1, p1 = model(img1)
            feature2, p2 = model(img2)
            cos_sim = F.cosine_similarity(feature1, feature2)
            cos_sim = cos_sim.detach().cpu().numpy()
            if(cos_sim > thrd):
                pred = 1
            else:
                pred = 0
            if pred == label:
                correct+=1
            del img1, img2, feature1, feature2, p1, p2
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

query_path = f'./queries.csv'
output_csv = f'./output/{output_filename}.csv'


test_dataset = TestDataset(query_path, test_dir)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle = False,
    drop_last=True
)

model.eval()
output = pd.DataFrame(columns= ['query', 'prediction'])
with torch.no_grad():
    for img1, img2, query in tqdm(test_dataloader):
        img1, img2 = img1.cuda(), img2.cuda()
        feature1, p1= model(img1)
        feature2, p2 = model(img2)
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
