from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.utils as vutils

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree

# data science imports
import pandas as pd
import numpy as np
import csv

import cxr_dataset as CXR
import eval_model as E
import test_model as T
import val_model as V
import densenet as densenet
import datetime

start = time.time()

PATH_TO_IMAGES = "../images_resized/"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
NUM_LAYERS = 58
FREEZE_LAYERS = 0
DROP_RATE = 0.0
CHROMOSOME = [58, 0, 2, 0]
NUM_OF_EPOCHS = 1
BATCH_SIZE = 32
currentDT = datetime.datetime.now()
# try:
#     rmtree('results/')
# except BaseException:
#     pass  # directory doesn't yet exist, no need to clear it
# os.makedirs("results/")

# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Scale(224),
        # because scale doesn't always give 224 x 224, this ensures 224 x
        # 224
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# create train/val dataloaders
transformed_datasets = {}
transformed_datasets['train'] = CXR.CXRDataset(
    path_to_images=PATH_TO_IMAGES,
    fold='train',
    transform=data_transforms['train'])
transformed_datasets['val'] = CXR.CXRDataset(
    path_to_images=PATH_TO_IMAGES,
    fold='val',
    transform=data_transforms['val'])

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(
    transformed_datasets['train'],
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0)
dataloaders['val'] = torch.utils.data.DataLoader(
    transformed_datasets['val'],
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0)

# load model
# checkpoint_best = torch.load('results/full_retrain/checkpoint')
checkpoint_best = torch.load('results/genetic_retrain/checkpoint')
model = checkpoint_best['model']
epoch_loss = 0.02

# summary(model)
# get preds and AUCs on test fold
a = T.make_pred_multilabel(
    data_transforms, model, PATH_TO_IMAGES, epoch_loss, CHROMOSOME)

end = time.time()
duration = end-start
print('Testing complete in {:.0f}m {:.0f}s'.format(
        duration // 60, duration % 60))