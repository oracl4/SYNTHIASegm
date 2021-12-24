import os
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import DataLoader

import utils
from labels import Labels
from dataset import SYNTHIADataset

# Parameter
DATA_DIR = './dataset/SYNTHIA-SF/SEQ1/'

# Segmentation Classes
CLASSES = ['void', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
            'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
            'road_lines', 'other', 'road_works']

# Define the Dataset Folder
x_dir = os.path.join(DATA_DIR, 'RGBLeft')
y_dir = os.path.join(DATA_DIR, 'GTLeft')

# Create dataset object
dataset = SYNTHIADataset(x_dir, y_dir, classes=CLASSES)

# Get random sample
idx = random.randint(0, dataset.__len__())
image, mask = dataset[idx]

# Convert mask to single label
labelMask = utils.convertMask(mask)

# Colorize the labal
colorizer = Labels()
rgb_mask = colorizer.colorize(labelMask)

# Visualize the Image and Labels
utils.visualize(
    image=image, 
    rgb_mask=rgb_mask,
)