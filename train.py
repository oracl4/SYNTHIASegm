import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import utils
from dataset import SYNTHIADataset

# Parameter
TRAIN_DATA_DIR = './dataset/SYNTHIA-SF/SEQ1/'
VAL_DATA_DIR = './dataset/SYNTHIA-SF/SEQ2/'
TEST_DATA_DIR = './dataset/SYNTHIA-SF/SEQ3/'


# Define the Dataset Folder
x_train_dir = os.path.join(TRAIN_DATA_DIR, 'RGBLeft')
y_train_dir = os.path.join(TRAIN_DATA_DIR, 'GTLeft')

x_valid_dir = os.path.join(VAL_DATA_DIR, 'RGBLeft')
y_valid_dir = os.path.join(VAL_DATA_DIR, 'GTLeft')

x_test_dir = os.path.join(TEST_DATA_DIR, 'RGBLeft')
y_test_dir = os.path.join(TEST_DATA_DIR, 'GTLeft')

# Augmentation Parameter
input_width = 640
input_height = 640

# Model Parameter
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'

CLASSES = ['void', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
            'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
            'road_lines', 'other', 'road_works']

# CLASSES = ['car']

ACTIVATION = 'softmax'
DEVICE = 'cuda'

# Testing
# trainDataset = SYNTHIADataset(images_dir=x_train_dir,
#                                 masks_dir=y_train_dir,
#                                 classes=['car'])

# image, mask = trainDataset[1]

# utils.visualize(
#     image=image, 
#     cars_mask=mask.squeeze(),
# )

# Augmentations
trainAug = albu.Compose([
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

valAug = albu.Compose([albu.PadIfNeeded(1088, 1920)])

# augmented_dataset = SYNTHIADataset(
#     images_dir=x_train_dir, 
#     masks_dir=y_train_dir, 
#     augmentation=trainAug, 
#     classes=['car'],
# )

# for i in range(3):
#     image, mask = augmented_dataset[1]
#     utils.visualize(image=image, mask=mask.squeeze(-1))

# create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = SYNTHIADataset(
    images_dir=x_train_dir, 
    masks_dir=y_train_dir, 
    augmentation=trainAug, 
    preprocessing=utils.get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = SYNTHIADataset(
    images_dir=x_train_dir, 
    masks_dir=y_train_dir, 
    augmentation=valAug, 
    preprocessing=utils.get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

max_score = 0

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')