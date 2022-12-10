# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 22:37:21 2022

@author: divshah
"""

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold


from constants import *

class MVTEC_AD_DATASET():
    
    def __init__(self, root):
        self.classes = ["good", "bad"] if NEG_CLASS ==1 else ["bad", "good"]
        self.img_transform = transforms.Compose(
            [transforms.Resize(INPUT_IMG_SIZE),
             transforms.ToTensor()])
        (self.img_filenames, 
         self.img_labels, 
         self.img_labels_detailed) = self._get_images_and_labels(root)
        
    def _get_images_and_labels(self, root):
        image_names = []
        labels = []
        labels_detailed = []
        
        for folder in DATASET_SETS:
            
            folder = os.path.join(root, folder)
            for class_folder in os.listdir(folder):
                label = (
                    1 - NEG_CLASS if class_folder == "good" else NEG_CLASS )
                
                label_detailed = class_folder
                
                class_folder = os.path.join(folder, class_folder)
                class_images = os.listdir(class_folder)
                class_images = [
                    os.path.join(class_folder, image)
                    for image in os.listdir(class_folder)
                    if image.find(IMG_FORMAT) > -1]
                
                """
                The extend method can add several elements to a list as opposed to append
                which can add only a single element at a time
                
                In this case class_images is a list containing the path to all the images
                
                """
                image_names.extend(class_images) 
                
                """
                For maintaining the list of labels, we find out the label for each
                sub-folder for example "good" or "anomaly", etc and then just extend the 
                labels list by the number of images in that folder
                """
                labels.extend([label] * len(class_images))
                labels_detailed.extend([label_detailed] * len(class_images))
                
        print("Dataset {}: N Images = {}, Share of anomalies = {:.3f}".format(
            root, len(labels), np.sum(labels)/len(labels)))

        return image_names, labels, labels_detailed

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_filenames[idx]  
        label = self.img_labels[idx]
        img = Image.open(img_path)
        img = self.img_transform(img)
        label = torch.as_tensor(label, dtype = torch.long)
        return img, label

def get_train_test_loaders(root, batch_size, test_size = 0.2, random_state = 10):
    dataset = MVTEC_AD_DATASET(root = root)
    """generating a list of indices for test and training data based on our collection
    
    Stratification is turned on to ensure there is equal distribution of all defect
    classes in test and train data
    """
    train_idx, test_idx = train_test_split(
        np.arange(dataset.__len__()),
        test_size = test_size,
        shuffle = True, 
        stratify= dataset.img_labels_detailed,
        random_state = random_state)
    
    train_sampler =  SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(
        dataset, batch_size = batch_size, sampler=train_sampler, drop_last=False)
    test_loader = DataLoader(
        dataset, batch_size = batch_size, sampler=test_sampler, drop_last=False)
    
    return train_loader, test_loader
    
         
        
        
