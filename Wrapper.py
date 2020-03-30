from Codes.FaceSwap_Tri import OneFaceInVideoAndImage, TwoFacesInVideo
from PRNet.demo_texture import texture_editing
import argparse
import cv2

# import numpy as np


def main():
	# print("yyyyyyyy")
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--video')          
	Parser.add_argument('--method')

	Args = Parser.parse_args()
	video = int(Args.video)
	method = int(Args.method)
	# print(video)
	if video ==0 and method == 1:
		cap = cv2.VideoCapture('Data/Data1.avi')
		OneFaceInVideoAndImage(cap)
	if video ==1 and method == 1:
		cap = cv2.VideoCapture('Data/Data2.mp4')
		# print("xxxxxxxxxxxxxxxxxxxx")
		TwoFacesInVideo(cap)

	if video == 





if __name__ == '__main__':
	main()