import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset as BaseDataset

class SYNTHIADataset(BaseDataset):
    """SYNTHIA-SF Segmentation Dataset.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """

    CLASSES = ['void', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
                'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
                'road_lines', 'other', 'road_works']
                
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        # Get the image and label list
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # Get the class that want to be detected
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        print("CustomDataset Created | Found %d Images" % (len(self.ids)))
    
    def __getitem__(self, i):
        """Get item from the dataset, return image and label in tensor
        Args:
            idx (int) : Index of the image
        """
        
        # Read the image from image list based on index
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read the label from label list based on index
        mask = cv2.imread(self.masks_fps[i])    # Get the BGR image
        mask = mask[..., 2]                     # Slice the list to take only the red channel image
        
        # Extract class from the mask based on the class that defined in the dataset
        # and perform one hot encoding to create N-channel image (N = number of class)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # Apply augmentation that defined in init
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing that defined in init
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
        
    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.ids)