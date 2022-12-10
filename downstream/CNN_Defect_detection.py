# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:54:38 2022

@author: divshah

This code implements a CNN based apporach for defect classification and detection
"""
#importing all constants
from constants import*
from data_loader_CNN import get_train_test_loaders
import os
from helper import train, evaluate, predict_localize
from model import CustomVGG
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

batch_size = 10

data_folder = os.path.join(root_dir, "bottle")
print(data_folder)

target_train_accuracy = 0.98
lr = 0.0001
epochs = 10
class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heatmap_thres = 0.7
n_cv_folds = 5

#data loading part

train_loader, test_loader = get_train_test_loaders(
    data_folder, batch_size, test_size=0.2, random_state=42)

path_saved_model = "./onnx_pytorch"
#model = CustomVGG()
model = torch.load(path_saved_model)

for param in model.parameters():
    param.requires_grad = False
#summary(model, input_size=(224,224,3))


"""
for values in model.state_dict():

  print(values, "\t", model.state_dict()[values].size())
"""

downstream_model = torch.nn.Sequential(
    model,
    nn.Linear(2048, 512),
    nn.Linear(512,2),
    nn.Sigmoid()
)

#summary(downstream_model, (224,224,3))

#class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model = train(
    train_loader, model, optimizer, criterion, epochs, device, target_train_accuracy
)

#evaluate(model, test_loader, device)
