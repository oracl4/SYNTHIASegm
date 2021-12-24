import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import wandb

import utils
from dataset import SYNTHIADataset

# Init wandb logs for training history
wandb.init(project="SYNTHIA-SF-SEGM")

# Dataset path
TRAIN_DATA_DIR = './dataset/SYNTHIA-SF/SEQ1/'
VAL_DATA_DIR = './dataset/SYNTHIA-SF/SEQ2/'

# Define the Dataset Folder
x_train_dir = os.path.join(TRAIN_DATA_DIR, 'RGBLeft')
y_train_dir = os.path.join(TRAIN_DATA_DIR, 'GTLeft')

x_valid_dir = os.path.join(VAL_DATA_DIR, 'RGBLeft')
y_valid_dir = os.path.join(VAL_DATA_DIR, 'GTLeft')

# Augmentation Parameter for Training
input_width = 640
input_height = 640

# Model Parameter
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax'
DEVICE = 'cuda'
LR = 1e-4
BS = 8

# Segmentation Classes
CLASSES = ['void', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
            'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
            'road_lines', 'other', 'road_works']

# Train Augmentations Pipeline
train_aug_pipeline = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    albu.PadIfNeeded(min_height=input_width, min_width=input_height, always_apply=True, border_mode=0),
    albu.RandomCrop(height=input_width, width=input_height, always_apply=True),
    albu.GaussNoise(p=0.2),
    albu.Perspective(p=0.5),
    albu.OneOf([
            albu.CLAHE(p=1),
            albu.RandomBrightnessContrast(contrast_limit=0, p=1),
            albu.RandomGamma(p=1),
            ], p=0.9,
    ),
    albu.OneOf([
            albu.Sharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
            ], p=0.9,
    ),
    albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0, p=1),
            albu.HueSaturationValue(p=1),
            ], p=0.9,
    ),
])

# Validation Augmentations Pipeline
val_aug_pipeline = albu.Compose([albu.PadIfNeeded(1088, 1920)])

# Create the segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

# Get the preprocessing function from the encoders
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Train dataset object
train_dataset = SYNTHIADataset(
    images_dir=x_train_dir, 
    masks_dir=y_train_dir, 
    augmentation=train_aug_pipeline, 
    preprocessing=utils.get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# Validation dataset object
valid_dataset = SYNTHIADataset(
    images_dir=x_valid_dir, 
    masks_dir=y_valid_dir, 
    augmentation=val_aug_pipeline, 
    preprocessing=utils.get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# Build the dataloader
train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=1, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

# Loss function that we used
loss = smp.utils.losses.DiceLoss()

# Metrics for performance
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# Optimizer
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=LR),
])

# Trainer object
trainer = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

# Validator object
validator = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# Maximum score for keeping records
max_score = 0

# Training loop
for i in range(0, 40):

    print('\nEpoch: {}'.format(i))

    # Train and validation
    train_logs = trainer.run(train_loader)
    valid_logs = validator.run(valid_loader)

    print("train:", train_logs)
    print("valid:", valid_logs)

    # Logs the loss and the iou score
    wandb.log({'train/dice_loss': train_logs['dice_loss'], 'train/iou_score': train_logs['iou_score']})
    wandb.log({'val/dice_loss': valid_logs['dice_loss'], 'val/iou_score': valid_logs['iou_score']})

    # If the validation score is higher than the max score > save the model
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './output/best_model.pth')
        print('Model saved!')
    
    # Reduce the learning rate on Epoch 25 to 0.00001
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')