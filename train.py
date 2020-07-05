import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import datetime
import cv2

from utils.accuracy import *
from utils.logger import Logger

optimizer = "SGD"
lr_rate = 0.045
mt = 0.9
wd = 0.00004
step = 30
g = 0.98
epoch_num = 200


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256,400)),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256,400)),
        # transforms.CenterCrop(256),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/hdd/stonehye/shot_data/'

path = {x: os.path.join(os.path.dirname(os.path.abspath(__file__)),data_dir,x)
                for x in ['train', 'val']}

image_datasets = {x: datasets.ImageFolder(path[x],
                                          data_transforms[x])
                  for x in ['train', 'val']}


dataloaders = { 'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=32,
                                             shuffle=True, num_workers=0),
                'val' : torch.utils.data.DataLoader(image_datasets['val'], batch_size=32//2,
                                             shuffle=True, num_workers=0) }
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}



class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

model_ft = models.mobilenet_v2(pretrained=True)
model_ft.classifier[1] = nn.Linear(model_ft.last_channel, 2)
# model_ft = model_ft.to(device)
if torch.cuda.is_available():
    model_ft = model_ft.cuda()
    model_ft = nn.DataParallel(model_ft)

# Loss function and Optimizer #
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr_rate, momentum=mt, weight_decay=wd )
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step, gamma=g)

# weight save option #
model_save_dir = '/hdd/stonehye/shot_data/models/' + "{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
if not os.path.isdir(model_save_dir):
    os.makedirs(model_save_dir)

losses = AverageMeter()
top1 = AverageMeter()
# top5 = AverageMeter()
losses.reset()
top1.reset()
# top5.reset()

log = Logger(directory=model_save_dir) # Create log file
log.line()
log(data_transforms) # 1. data transform info
log.line()
log(model_ft) # 2. network layer info
log.line()

model_save_point = 3

# train #
for epoch in range(epoch_num):
    log.time()
    log('Epoch\t{}'.format(epoch))
    exp_lr_scheduler.step()

    # Save
    # if epoch % model_save_point == 0:
    #     model_save_path = os.path.join(model_save_dir, "{:05}.pth".format(epoch))
    #     torch.save(model_ft.state_dict(), model_save_path)
    #     log("Save model\t{}".format(model_save_path))

    # train
    model_ft.train()
    losses.reset()
    top1.reset()

    for batch_idx, (data, target) in enumerate(dataloaders['train']):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer_ft.zero_grad()
        output = model_ft.forward(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer_ft.step()

        losses.update(loss.item(), target.size(0))
        log("Loss @{}\t{}".format(batch_idx, loss.item()))

        acc1, acc5 = accuracy(output, target, topk=(1, 1))
        top1.update(acc1[0], target.size(0))

    log('Train Loss\t{}'.format(losses.avg))
    log('Train Top1 Accuracy\t{}'.format(top1.avg))

    # validation
    model_ft.eval()
    top1.reset()

    for batch_idx, (data, target) in enumerate(dataloaders['val']):
        print("{} test".format(batch_idx))
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = model_ft.forward(data)
        acc1, acc5 = accuracy(output, target, topk=(1,1))
        top1.update(acc1[0], target.size(0))

    log('Validation Top1 Accuracy\t{}'.format(top1.avg))
    log.line()

    if epoch % model_save_point == 0:
        log.time()
        log('Epoch\t{}'.format(epoch))
        model_save_path = os.path.join(model_save_dir, "{:05}.pth".format(epoch))
        torch.save(model_ft.state_dict(), model_save_path)
        log("Save model\t{}".format(model_save_path))
        log.line()


