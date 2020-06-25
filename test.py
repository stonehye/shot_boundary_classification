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

model_path = '/hdd/stonehye/shot_data/models/20200625055718/00210.pth'
testset_path = '/hdd/stonehye/shot_data/test/'

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder(testset_path, transform=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32//2, shuffle=True, num_workers=0)
testset_size = len(test_dataset)
class_names = test_dataset.classes

m = models.mobilenet_v2(pretrained=False)
m.classifier[1] = nn.Linear(m.last_channel, 2)
m = m.cuda()
m = nn.DataParallel(m)
m.load_state_dict(torch.load(model_path))

top1 = AverageMeter()
top1.reset()

total = 0
correct = 0
TP = 0
FP = 0
FN = 0
TN = 0
# save_num = 0
for batch_idx, (data, target) in enumerate(testloader):
    # print("{} test".format(batch_idx))
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    output = m.forward(data)
    acc1, acc5 = accuracy(output, target, topk=(1,1))
    top1.update(acc1[0], target.size(0))

    _, predicted = torch.max(output.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()

    # 오답 이미지 저장 #
    # for idx, image in enumerate(data):
    #     if (predicted[idx]!=target[idx]):
    #         if predicted[idx].item() == 0:
    #             torchvision.utils.save_image(image,'./wrong/positive/'+str(save_num)+".jpg")
    #         else:
    #             torchvision.utils.save_image(image,'./wrong/negative/'+str(save_num)+".jpg")
    #         save_num+=1

    for idx, image in enumerate(data):
        if (predicted[idx]!=target[idx]): # wrong output
            if predicted[idx].item() == 0:
                FN += 1
            else:
                FP += 1
        else: # correct output
            if predicted[idx].item() == 0:
                TN += 1
            else:
                TP += 1


print("Top1 Accuraccy\t{}".format(top1.avg))
# print('Accuracy: %d %%' % (100 * correct / total))
Pre = TP/(TP+FP)
Rec = TP/(TP+FN)
print("Precision\t{}".format(Pre))
print("Recall\t{}".format(Rec))
print("F1 Score\t{}".format(2*(Pre*Rec/(Pre+Rec))))
