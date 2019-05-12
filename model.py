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
from tensorboardX import SummaryWriter
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
import val_model as V
import densenet as densenet
import datetime

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
date = datetime.date.today()
f = open(f"logs/output-retrain-{date}.txt", "a+")
print("Available GPU count:" + str(gpu_count),file=f)


def checkpoint(model, best_loss, epoch, LR):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving',file=f)
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, 'results/checkpoint')


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,
        PATH_TO_IMAGES,
        CHROMOSOME,
        data_transforms):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs),file=f)
        print('-' * 10,file=f)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            print("train "+ str(len(dataloaders['train'])),file=f)
            print("val " + str(len(dataloaders['val'])),file=f)
            dataloader_train = iter(dataloaders[phase])
            # for data in dataloaders[phase]: 
            for i in range(len(dataloaders[phase])): 
                start = time.time()
                data = next(dataloader_train)
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * batch_size
                end = time.time()
                execution = end-start
                # print(f"iteration : {i}")
                # print(f"running_loss : {running_loss}")
                # print(execution)

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]),file=f)

            # decay learning rate if no val loss improvement in this epoch

            # old decay learning rate
            # if phase == 'val' and epoch_loss > best_loss:
            #     print("decay loss from " + str(LR) + " to " +
            #           str(LR / 10) + " as not seeing improvement in val loss")
            #     LR = LR / 10

            if phase == 'val' and epoch % 30 == 0:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " at every 30 epochs",file=f)
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR),file=f)

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

                    # TODO tambahin auc score buat tiap val
                    # preds, aucs = V.make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES, epoch_loss, CHROMOSOME)

            if phase == 'train':
                fi= open(f"logs/epoch_loss_train_{CHROMOSOME}.txt","a+")
                fi.write(f"{epoch},{epoch_loss}\n")
                fi.close()
            elif phase == 'val':
                f= open(f"logs/epoch_loss_val_{CHROMOSOME}.txt","a+")
                f.write(f"{epoch},{epoch_loss}\n")  
                f.close()

        # weight = model.features.conv0.weight
        # to_pil_image = transforms.ToPILImage()
        # # writer = SummaryWriter()
        # #TODO print image convolve 1
        # for x in range(64):
        #     # image = vutils.make_grid(weight[x], normalize=True, scale_each=True)
        #     # writer.add_image('Image', image, x)
        #     img = to_pil_image(weight[x].cpu())
        #     image_path = 'images/'+str(epoch)+'/'+str(x)+'.png'
        #     img.save(image_path)

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch",file=f)

        # old stopper
        # break if no val loss improvement in 3 epochs
        # if ((epoch - best_epoch) >= 3):
        #     print("no improvement in 3 epochs, break")
        #     break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60),file=f)

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch, epoch_loss


def train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY, NUM_LAYERS, FREEZE_LAYERS, DROP_RATE, CHROMOSOME, NUM_OF_EPOCHS):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = NUM_OF_EPOCHS
    BATCH_SIZE = 64
    currentDT = datetime.datetime.now()
    # try:
    #     rmtree('results/')
    # except BaseException:
    #     pass  # directory doesn't yet exist, no need to clear it
    # os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
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

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
       raise ValueError("Error, requires GPU")

    print("=> Training using ",DROP_RATE," drop rate",file=f)
    print("=> Training using ",LR," learning rate",file=f)
    print("=> Use ",NUM_LAYERS," layers in blocks",file=f)

    model = densenet.densenet121(pretrained=True, num_layers=NUM_LAYERS, drop_rate=DROP_RATE)
    
    #freezing layers
    print("=> Freezing ",FREEZE_LAYERS," layers in blocks",file=f)

    if FREEZE_LAYERS > 42:
        freeze_transition = 3
    elif FREEZE_LAYERS > 18:
        freeze_transition = 2
    elif FREEZE_LAYERS > 6:
        freeze_transition = 1
    elif FREEZE_LAYERS >= 0:
        freeze_transition = 0

    i=0
    limit_freeze = 0
    #add freeze for freeze layers
    limit_freeze+=(FREEZE_LAYERS * 6)
    #add freeze for all transition layer that been pass through
    limit_freeze+=(freeze_transition*3)
    #add freeze for first convolution
    if limit_freeze > 0:
        limit_freeze+=3 
    print(limit_freeze,file=f)
    for name, param in model.named_parameters():
        if i< limit_freeze:
            # print(name,file=f)
            if("sEBlock" in name):
                print("skipping layer : "+name,file=f)
            else:
                param.requires_grad = False
                i=i+1
   
    # print model
    # print("MODEL",file=f)
    # print(model,file=f)

    #print grad
    # print("GRAD",file=f)
    # # print(model.features.pool0,file=f)
    # for name,param in model.named_parameters():
    #     print (name,file=f)
    #     print(param.requires_grad,file=f)

    num_ftrs = model.classifier.in_features
    # add final layer with # outputs in same dimension of labels with sigmoid
    # activation
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    # optimizer = optim.Adadelta(
    #     filter(
    #         lambda p: p.requires_grad,
    #         model.parameters()),
    #     lr=LR,
    #     weight_decay=WEIGHT_DECAY)

    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # train model
    model, best_epoch, epoch_loss = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,
                                     PATH_TO_IMAGES=PATH_TO_IMAGES, CHROMOSOME = CHROMOSOME, data_transforms = data_transforms)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES, epoch_loss, CHROMOSOME)

    return preds, aucs, epoch_loss
