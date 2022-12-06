# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 23:46:27 2022

@author: Divyaj
"""

wdir="D:\CS230_Deep_Learning\Projects\SimCLR_Defect_Detection"
dataset_name="stl10"
IMG_SIZE = (224,224)
INPUT_SHAPE = IMG_SIZE + (3,)
# Algorithm hyperparameters
batch_size = 32
num_epochs = 50
temperature = 0.1
width = 128
# Stronger augmentations for contrastive, weaker ones for supervised training
contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
classification_augmentation = {"min_area": 0.75, "brightness": 0.3, "jitter": 0.1}
cut_paste_augmentation = {"x_scale":10, "y_scale":10, "IMG_SIZE":IMG_SIZE}
# Dataset hyperparameters
unlabeled_dataset_size = 1000
labeled_dataset_size = 50
