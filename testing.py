import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import utils
from dataset import SYNTHIADataset

# Dataset path
TEST_DATA_DIRS = [
    './dataset/SYNTHIA-SF/SEQ1/',
    './dataset/SYNTHIA-SF/SEQ2/',
    './dataset/SYNTHIA-SF/SEQ3/',
    './dataset/SYNTHIA-SF/SEQ4/',
    './dataset/SYNTHIA-SF/SEQ5/',
    './dataset/SYNTHIA-SF/SEQ6/'
]

# Segmentation Classes
CLASSES = ['void', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
            'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
            'road_lines', 'other', 'road_works']

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

# Load the best model checkpoint
model = torch.load('output/best_model.pth')

# Validation Augmentations Pipeline
val_aug_pipeline = albu.Compose([albu.PadIfNeeded(1088, 1920)])

# Get the preprocessing function from the encoders
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Loss function that we used
loss = smp.utils.losses.DiceLoss()

# Metrics for performance
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# Loop through the dataset
for TEST_DATA_DIR in TEST_DATA_DIRS:

    # Define the dataset Folder
    x_dir = os.path.join(TEST_DATA_DIR, 'RGBLeft')
    y_dir = os.path.join(TEST_DATA_DIR, 'GTLeft')

    # Create the dataset object
    test_dataset = SYNTHIADataset(
        x_dir, 
        y_dir, 
        augmentation=val_aug_pipeline, 
        preprocessing=utils.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataloader = DataLoader(test_dataset)

    # Evaluate the model on the dataset
    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)

    print(logs)