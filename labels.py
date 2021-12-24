from collections import namedtuple

import numpy as np

class Labels:

    Label_entry = namedtuple("Label_entry", ['name', 'color'],)

    """
    The total number of classes in the training set.
    """
    NUM_CLASSES = 23
    
    """
    ID is equal to index in this array.
    """
    COLOURS = [
        Label_entry('void', (0, 0, 0)),                 # ID = 0
        Label_entry('road', (128, 64, 128)),            # ID = 1
        Label_entry('sidewalk', (244, 35, 232)),        # ID = 2
        Label_entry('building', (70, 70, 70)),          # ID = 3
        Label_entry('wall', (102, 102, 156)),           # ID = 4
        Label_entry('fence', (190, 153, 153)),          # ID = 5
        Label_entry('pole', (153, 153, 153)),           # ID = 6
        Label_entry('traffic light', (250, 170, 30)),   # ID = 7
        Label_entry('traffic sign', (220, 220, 0)),     # ID = 8
        Label_entry('vegetation', (107, 142, 35)),      # ID = 9
        Label_entry('terrain', (152, 251, 152)),        # ID = 10   
        Label_entry('sky', (70, 130, 180)),             # ID = 11
        Label_entry('person', (220, 20, 60)),           # ID = 12
        Label_entry('rider', (255, 0, 0)),              # ID = 13
        Label_entry('car', (0, 0, 142)),                # ID = 14
        Label_entry('truck', (0, 0, 70)),               # ID = 15
        Label_entry('bus', (0, 60, 100)),               # ID = 16
        Label_entry('train', (0, 80, 100)),             # ID = 17
        Label_entry('motorcycle', (0, 0, 230)),         # ID = 18
        Label_entry('bicycle', (119, 11, 32)),          # ID = 19
        Label_entry('road_lines', (157, 234, 50)),      # ID = 20
        Label_entry('other', (72, 0, 98)),              # ID = 21
        Label_entry('road_works', (167, 106, 29))       # ID = 22
    ]

    """
    Label is numpy array object!
    """

    @staticmethod
    def colorize(label):
        
        #Unlabelled pixels are black (zero)!
        colorized = np.zeros((label.shape[0], label.shape[1], 3), dtype = np.uint8)

        for idx in range(len(Labels.COLOURS)):
            colorized[idx == label] = Labels.COLOURS[idx].color

        return colorized  # Numpy array object!