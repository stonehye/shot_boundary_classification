from findframe import *
import cv2
import glob
import math
import numpy as np
import sys

result_dir = '/hdd/stonehye/shot_data/video_data/trecvid2018/'

def SBD(videopath='1e5aec100a5b897342a4be9246995fc16f784e66.flv'):
	# dataload
	# videopath = '1e5aec100a5b897342a4be9246995fc16f784e66.flv'
	cap = cv2.VideoCapture(str(videopath))
	curr_frame = None
	prev_frame = None
	frame_diffs = []
	frames = []
	success, frame = cap.read()
	frameId = cap.get(1)
	frameRate = cap.get(5)
	i = 0
	FRAME = Frame(0, 0)
	while (success):
		"""
		calculate the difference between frames 
		"""
		if (frameId % math.floor(frameRate) == 0):
			luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
			curr_frame = luv
			if curr_frame is not None and prev_frame is not None:
				diff = cv2.absdiff(curr_frame, prev_frame)
				diff_sum = np.sum(diff)
				diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
				frame_diffs.append(diff_sum_mean)
				frame = Frame(i, diff_sum_mean)
				frames.append(frame)
			elif curr_frame is not None and prev_frame is None:
				diff_sum_mean = 0
				frame_diffs.append(diff_sum_mean)
				frame = Frame(i, diff_sum_mean)
				frames.append(frame)

			prev_frame = curr_frame
			i = i + 1
		success, frame = cap.read()
		frameId = cap.get(1)
	cap.release()

	# detect the possible frame
	frame_return, start_id_spot_old, end_id_spot_old = FRAME.find_possible_frame(frames)

	# optimize the possible frame
	new_frame, start_id_spot, end_id_spot = FRAME.optimize_frame(frame_return, frames)

	# store the result
	start = np.array(start_id_spot[1:])[np.newaxis, :]
	end = np.array(end_id_spot[:-1])[np.newaxis, :]
	spot = np.concatenate((end.T, start.T), axis=1)
	np.savetxt(result_dir+videopath.split('/')[-1]+'.txt', spot, fmt='%d', delimiter='\t')


if __name__ == '__main__':
	'''
	Detect shot boundary frame num
	'''
	video_list = glob.glob('/hdd/stonehye/shot_data/video_data/trecvid2018/*')
	video_list = [file for file in video_list if (file.endswith('.flv') or file.endswith('.mp4'))]

	c = 0
	for video_path in video_list:
		c+=1
		print(video_path + "   %d/%d" % (c, len(video_list)))
		try:
			SBD(videopath=video_path)
		except:
			pass