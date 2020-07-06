import os
import cv2
import glob
import numpy as np
import random

from data_augmentation import add_blackborder

BLACKBORDER = 1
pair_list = list()


def concatenate(frame1, frame2, output_name = './result.jpg'):
    # if you want to add blackborder (for augmentation)
    if BLACKBORDER:
        Top = int(random.choice([2,3,4,5,6])*0.01 * frame1.shape[0])
        Bottom = Top
        Left = random.choice([0, Top])
        Right = Left
        frame1 = add_blackborder(top=Top, bottom=Bottom, left=Left, right =Right, img=frame1)
        frame2 = add_blackborder(top=Top, bottom=Bottom, left=Left, right =Right, img=frame2)

    height, width = frame1.shape[:2]
    frame1_cropped = frame1[0:height, 0:int(width/2)]
    height, width = frame2.shape[:2]
    frame2_cropped = frame2[0:height, 0:int(width/2)]
    result = cv2.hconcat((frame1_cropped, frame2_cropped))
    cv2.imwrite(output_name, result)


if __name__ == '__main__':
    directory = '/hdd/stonehye/shot_data/non_shot_boundary_frames_complete/*'
    # directory = '/hdd/stonehye/shot_data/*'
    result_directory = '/hdd/stonehye/shot_data/train/negative/'
    frame_list = glob.glob(directory)
    frame_list = [file for file in frame_list if (file.endswith('.jpg'))]
    frame_list = sorted(frame_list)
    
    i=0
    # for i in range(0,len(frame_list),2):
    while i<len(frame_list):
        # valid check
        temp = frame_list[i].split('/')[-1].split('_')
        img1_data = '_'.join(temp[0:-1])
        img1_num = int(temp[-1].replace('.jpg', ''))
        temp = frame_list[i+1].split('/')[-1].split('_')
        img2_data = '_'.join(temp[0:-1])
        img2_num = int(temp[-1].replace('.jpg', ''))
        
        if img1_data == img2_data and abs(img1_num-img2_num) == 1:
            img1 = cv2.imread(frame_list[i])
            img2 = cv2.imread(frame_list[i+1])
            pair_list.append((img1, img2))
            i+=2
        else:
            next_file = '_'.join(frame_list[i].split('_')[0:-1]) + '_' + str(img1_num+1) + '.jpg'
            if os.path.isfile(next_file):
                temp = next_file.split('/')[-1].split('_')
                img2_data = ''.join(temp[0:-1])
                img2_num = int(temp[-1].replace('.jpg', ''))
                pair_list.append((img1, img2))
            else:
                print(frame_list[i])
            i+=1
    
    cnt = 0
    for i in pair_list:
        concatenate(i[0], i[1], output_name=result_directory+str(cnt)+'_border.jpg')
        cnt += 1

