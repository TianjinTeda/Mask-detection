from __future__ import print_function, division

import torch
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils
import pylab

# Set the device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create the data loader
data_dir = 'data/mask'
mask_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
mask_data_loader = {x: torch.utils.data.DataLoader(mask_dataset[x], batch_size=64, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(mask_dataset[x]) for x in ['train', 'val']}
class_names = mask_dataset['train'].classes

if __name__ == '__main__':

    # Print device information
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())

    # Print the dataset information
    print(dataset_sizes)
    print(class_names)

    # Inspect the dataset images
    image_batch = next(iter(mask_data_loader['train']))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(image_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    pylab.show()
    print("The classes are: ")
    print(image_batch[1])