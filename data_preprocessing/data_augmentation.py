import sys
import glob
import os
import cv2
import random

def add_blackborder(top=0, bottom=0, left=0, right=0, img=None):
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, [0,0,0])


def main():
    img = cv2.imread("test.jpg")
    Top = int(random.choice([2,3,4,5,6])*0.01 * img.shape[0])
    Bottom = Top
    Left = Top
    Right = Top
    img_result = add_blackborder(top=Top, bottom=Bottom, left=Left, right =Right, img=img)
    cv2.imwrite("test_result.jpg", img_result)