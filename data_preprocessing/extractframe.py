import cv2
import glob
import numpy as np
import sys
from random import *

result_dir = '/hdd/stonehye/shot_data/temp_negative/'
# result_dir = './'
negative_max = 750


def Frame(videopath, frame_list):
	cap = cv2.VideoCapture(str(videopath))
	success, frame = cap.read()
	filename = videopath.split('/')[-1]
	i = 0
	while (success):
		if str(i) in frame_list:
			cv2.imwrite(result_dir+filename+'_'+str(i)+'.jpg', frame)
			# print(result_dir+filename+'_'+str(i)+'.jpg')
		i = i + 1
		success, frame = cap.read()
	cap.release()


def positive():
	video_list = glob.glob('/hdd/stonehye/shot_data/video_data/vcdb_core/*')
	video_list = [file for file in video_list if (file.endswith('.flv') or file.endswith('.mp4'))]
	# print(video_list)

	for file in video_list:
		try:
			frame_list = list()
			text = file+'.txt'
			f = open(text, 'r')
			lines = f.readlines()
			for line in lines:
				line = line.strip('\n')
				frame_list += line.split('\t')
			frame_list = sorted(list(set(frame_list)), key=int)
			Frame(file, frame_list)
			f.close()
		except:
			print(file)


def negative():
	video_list = glob.glob('/hdd/stonehye/shot_data/video_data/vcdb_core/*')
	video_list = [file for file in video_list if (file.endswith('.flv') or file.endswith('.mp4'))]
	negative_cnt = 1

	for file in video_list:
		try:
			frame_list = list()
			text = file+'.txt'
			f = open(text, 'r')
			lines = f.readlines()
			prev = 0
			for line in lines:
				line = line.strip('\n').split('\t')
				curr = int(line[0])
				try:
					i = randint(prev+1, curr-1)
					frame_list += [str(i), str(i+1)]
					negative_cnt += 1
				except:
					# print("wrong range")
					# print(prev+1, curr-1)
					pass
				finally:
					prev = int(line[1])
			frame_list = sorted(list(set(frame_list)), key=int)
			Frame(file, frame_list)
			f.close()
		except:
			print(file)
		finally:
			if negative_cnt >= negative_max:
				break
	print("total data: ", negative_cnt)


if __name__ == '__main__':
	negative()
			