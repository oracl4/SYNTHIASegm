import os
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu

def visualize(**images):
    """PLot images in one row"""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def convertMask(mask):
    """Convert one hot encoded mask to single array with label"""
    n_class = len(mask[1, 1, :])
    
    for i in range(n_class):
        class_mask = mask[:,:,i]
        mask[class_mask == 1.0] = i

    array = mask[:, :, 1]

    return array

def to_tensor(x, **kwargs):
    """Convert image to tensor"""
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)