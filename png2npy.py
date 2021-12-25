import numpy as np
from PIL import Image
import os
import argparse
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from torchvision.io import  ImageReadMode

parser = argparse.ArgumentParser()
parser.add_argument("--path", default='./')
parser.add_argument("--filename")
args = parser.parse_args()

file_path = f'{args.path}{args.filename}/'
img_dir = os.listdir(file_path)
# print(img_dir)
print(len(os.listdir(file_path)))

for imgname in tqdm(img_dir):
    if not os.path.exists(f'{args.filename}_npy'):
        os.mkdir(f'{args.filename}_npy')
    image_full_dir = os.path.join(file_path, imgname)
    img = Image.open(image_full_dir)
    data = np.array( img, dtype='uint8' )
    # print(data.shape)
    # np.save( f'{args.path}/{args.filename}_npy/{imgname[:-4]}.npy', data)
    # print(f'{args.path}{args.filename}_npy/{imgname[:-4]}.npy')

print(len(os.listdir(f"{args.path}{args.filename}_npy")))
# visually testing our output
# img_array = np.load(filename + '.npy')
# plt.imshow(img_array)