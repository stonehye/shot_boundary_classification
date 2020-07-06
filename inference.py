import torch
import torch.nn as nn
import numpy as np
import torchvision
import os
import cv2
import sys
import time
import math
import argparse
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from PIL import Image

# model_path = 'model/00210.pth'
model_path = '/hdd/stonehye/shot_data/models/20200706012635/00198.pth'

m = models.mobilenet_v2(pretrained=False)
m.classifier[1] = nn.Linear(m.last_channel, 2)
m = m.cuda()
m = nn.DataParallel(m)
m.load_state_dict(torch.load(model_path))
m.eval()


def inference(image):
    # image = Image.open(image_path)
    data_transforms = transforms.Compose([
        transforms.Resize((256,400)),
        transforms.ToTensor()
    ])
    x = data_transforms(image)
    img1 = x[0]
    save_image(img1, "test.jpg")
    x.unsqueeze_(0)
    x.cuda()

    output = m(x)
    _, predicted = torch.max(output.data, 1)
    # print(predicted.item())
    return (predicted.item())


def crop_and_concat(frame1, frame2):
    height, width = frame1.shape[:2]
    frame1_cropped = frame1[0:height, 0:int(width/2)]
    height, width = frame2.shape[:2]
    frame2_cropped = frame2[0:height, 0:int(width/2)]
    result = cv2.hconcat((frame1_cropped, frame2_cropped))
    return result


def frame_extract(videopath):
    frame_list = list()
    cap = cv2.VideoCapture(str(videopath))
    frameRate = cap.get(5)
    success, frame = cap.read()
    i = 0
    while (success):
        frameId = cap.get(1)
        if (frameId % math.floor(frameRate) == 0):
            frame_list.append(frame)
            i = i + 1
        success, frame = cap.read()
    cap.release()
    return frame_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shot Boundary Classification Test")

    parser.add_argument('--option', type=str, default="image", help="file format to test")
    parser.add_argument('--img1', type=str, default="test_data/image1.jpg", help="First image path corresponding to shot boundary")
    parser.add_argument('--img2', type=str, default="test_data/image2.jpg", help="Second image path corresponding to shot boundary")
    parser.add_argument('--videopath', type=str, default="test_data/video.flv", help="Video path to detect shots")
    # parser.add_argument('--resulttxt', type=str, default="test/result.txt", help="Text file name to output the shot boundary frame number")
    
    args = parser.parse_args()
    # print(args)

    if args.option == "image" or args.option == "Image":
        print("test image path: {} and {}".format(args.img1, args.img2))
        img1 = cv2.imread(args.img1)
        img2 = cv2.imread(args.img2)
        test_image = crop_and_concat(img1, img2)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_image = Image.fromarray(test_image)
        result = inference(test_image)
        print("result: ", end='')
        if result:
            print("shot boundary")
        else:
            print("Non shot boundary")

    elif args.option == "video" or args.option == "Video":
        print("test video path: {}".format(args.videopath))
        videoname = args.videopath.split('/')[-1]
        frame_list = frame_extract(args.videopath)
        shot_boundary_list = list()
        directory_path = 'SB/'+ videoname + '/'
        os.makedirs(directory_path)
        for idx, frame in enumerate(frame_list[1:]):
            print("%.2f%%" %(100*(idx/len(frame_list[1:])))) # print percent
            idx += 1
            test_image = crop_and_concat(frame_list[idx-1], frame_list[idx])
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            test_image = Image.fromarray(test_image)
            result = inference(test_image)
            if result:
                # test_image.save('SB/'+str(idx)+'.jpg') #경계이미지 저장
                cv2.imwrite(directory_path+str(idx-1)+'.jpg', frame_list[idx-1])
                cv2.imwrite(directory_path+str(idx)+'.jpg', frame_list[idx])
                shot_boundary_list.append((idx-1, idx))
            sys.stdout.write("\033[F") # Cursor up one line
        print("%.2f%%" % 100)
        time.sleep(1)

        print("shot interval list")
        prev = 0
        for pair in shot_boundary_list:
            curr = pair[0]
            print("{}\t{}".format(prev, curr))
            prev = pair[1]
        print("{}\t{}".format(prev, len(frame_list)-1))