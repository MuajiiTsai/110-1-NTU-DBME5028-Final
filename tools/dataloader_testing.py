from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import transforms
import PIL
import numpy as np


plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        img.save("./ref_img/loadertest.png")
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        img.save


def dataloader_test(dataloader, num_of_image):
    if num_of_image == 2:
        features1, features2 = next(iter(dataloader))
        print(f"feature1 batch size: {features1.size()}")
        print(f"feature2 batch size: {features2.size()}")
        # print(f"labels batch size: {labels.shape}")
        print(f"dataloader size: {len(dataloader)}")

        features2 = features2.squeeze(0)
        features1 = features1.squeeze(0)

        # print(features2)
        # print(query)
        a = make_grid([features1, features2])
        show(a)
    elif num_of_image == 3:
        features1, features2, features3 = next(iter(dataloader))
        print(f"feature1 batch size: {features1.size()}")
        print(f"feature2 batch size: {features2.size()}")
        print(f"feature3 batch size: {features3.size()}")
        # print(f"labels batch size: {labels.shape}")
        print(f"dataloader size: {len(dataloader)}")

        features3 = features3.squeeze(0)
        features2 = features2.squeeze(0)
        features1 = features1.squeeze(0)

        # print(features2)
        # print(query)
        a = make_grid([features1, features2, features3])
        show(a)