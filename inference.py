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
from labels import Labels

# Dataset path
TEST_DATA_DIR = './dataset/SYNTHIA-SF/SEQ5/'

# Define the dataset Folder
x_dir = os.path.join(TEST_DATA_DIR, 'RGBLeft')
y_dir = os.path.join(TEST_DATA_DIR, 'GTLeft')

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

# Create the dataset object
test_dataset = SYNTHIADataset(
    x_dir, 
    y_dir, 
    augmentation=val_aug_pipeline, 
    preprocessing=utils.get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# Test dataset without transformation
test_dataset_vis = SYNTHIADataset(
    x_dir,
    y_dir, 
    classes=CLASSES,
)

# Colorize the labal
colorizer = Labels()

# Visualize 5 inference prediction
for i in range(5):
    
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    gt_transpose = gt_mask.transpose(1,2,0)
    pr_transpose = pr_mask.transpose(1,2,0)

    # Convert mask to single label
    gt_single = utils.convertMask(gt_transpose)
    pr_single = utils.convertMask(pr_transpose)

    gt_rgb = colorizer.colorize(gt_single)
    pr_rgb = colorizer.colorize(pr_single)    
        
    utils.visualize(
        image=image_vis, 
        ground_truth_mask=gt_rgb, 
        predicted_mask=pr_rgb
    )