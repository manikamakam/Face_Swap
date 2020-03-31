from Codes.FaceSwap_Tri import OneFaceInVideoAndImage, TwoFacesInVideo,shape_to_np
from PRNet.demo_texture import texture_editing
from PRNet.api import PRN
import argparse
import cv2
import os
import numpy as np
from copy import deepcopy
import dlib

def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--video')          
	Parser.add_argument('--method')

	Args = Parser.parse_args()
	video = int(Args.video)
	method = int(Args.method)

	if video ==0 and method == 1:
		cap = cv2.VideoCapture('Data/Data1.mp4')
		OneFaceInVideoAndImage(cap)
	if video ==1 and method == 1:
		cap = cv2.VideoCapture('Data/Data2.mp4')
		TwoFacesInVideo(cap)

	if video == 0 and method == 3:
		Parser.add_argument('-i', '--image_path', default='PRNet/TestImages/frame.jpg', type=str,
		                    help='path to input image')
		Parser.add_argument('-r', '--ref_path', default='TestSet_P2/Scarlett.jpg', type=str, 
		                    help='path to reference image(texture ref)')
		Parser.add_argument('-o', '--output_path', default='PRNet/TestImages/output.jpg', type=str, 
		                    help='path to save output')
		Parser.add_argument('--mode', default=1, type=int, 
		                    help='ways to edit texture. 0 for modifying parts, 1 for changing whole')
		Parser.add_argument('--gpu', default='0', type=str, 
			                        help='set gpu id, -1 for CPU')
		
		# ---- init PRN
		os.environ['CUDA_VISIBLE_DEVICES'] = Parser.parse_args().gpu # GPU number, -1 for CPU
		prn = PRN(is_dlib = True) 

		cap = cv2.VideoCapture('TestSet_P2/Test3.mp4')
		if (cap.isOpened() == False):
			print("Unable to read camera feed")

		out = cv2.VideoWriter('Data/TestSetOutputs/Test3OutputPRNet.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (int(cap.get(3)), int(cap.get(4))))
		img_array =[]
		while (True):
			ret, frame = cap.read()
			if ret == True:
				img = deepcopy(frame)

				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				detector = dlib.get_frontal_face_detector()
				predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
				rects = detector(img, 1)

				if len(rects) >0:	
					
					cv2.imwrite('PRNet/TestImages/frame.jpg', frame) 
					texture_editing(prn, Parser.parse_args())
					res = cv2.imread('PRNet/TestImages/output.jpg')
					img_array.append(res)
				else:
					img_array.append(frame)
			else:
				break
		print(len(img_array))
		for i in range(len(img_array)):
			out.write(img_array[i])
		out.release()
		cap.release()

	if video == 1 and method == 3:
		Parser.add_argument('-i', '--image_path', default='PRNet/TestImages/frame.jpg', type=str,
		                    help='path to input image')
		Parser.add_argument('-r', '--ref_path', default='PRNet/TestImages/frame2.jpg', type=str, 
		                    help='path to reference image(texture ref)')
		Parser.add_argument('-o', '--output_path', default='PRNet/TestImages/output.jpg', type=str, 
		                    help='path to save output')
		Parser.add_argument('--mode', default=1, type=int, 
		                    help='ways to edit texture. 0 for modifying parts, 1 for changing whole')
		Parser.add_argument('--gpu', default='0', type=str, 
			                        help='set gpu id, -1 for CPU')
		args = Parser.parse_args()
		# ---- init PRN
		os.environ['CUDA_VISIBLE_DEVICES'] = Parser.parse_args().gpu # GPU number, -1 for CPU
		prn = PRN(is_dlib = True) 

		cap = cv2.VideoCapture('TestSet_P2/Test2.mp4')
		if (cap.isOpened() == False):
			print("Unable to read camera feed")

		out = cv2.VideoWriter('Data/TestSetOutputs/Test2OutputPRNet.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (int(cap.get(3)), int(cap.get(4))))
		img_array =[]
		while (True):
			ret, frame = cap.read()
			if ret == True:
				img = deepcopy(frame)
				frame2 = deepcopy(frame)

				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				detector = dlib.get_frontal_face_detector()
				predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
				rects = detector(img, 1)

				if len(rects) >=2:	
					dst_pts = predictor(img, rects[0])
					dst_pts = shape_to_np(dst_pts)
					dst_pts2 = predictor(img, rects[1])
					dst_pts2 = shape_to_np(dst_pts2)

					avg1 = sum(dst_pts[:,0])/len(dst_pts[:,0])
					avg2 = sum(dst_pts2[:,0])/len(dst_pts2[:,0])

					avg = (avg1 +avg2)/2
					h,w,c = frame.shape
					frame[:,avg:w] = (0,0,0)
					frame2[:,0:avg] = (0,0,0)

					cv2.imwrite('PRNet/TestImages/frame.jpg', frame)
					cv2.imwrite('PRNet/TestImages/frame2.jpg', frame2) 
					args.image_path = 'PRNet/TestImages/frame.jpg'
					args.ref_path = 'PRNet/TestImages/frame2.jpg'

					texture_editing(prn, args)
					res1 = cv2.imread('PRNet/TestImages/output.jpg')

					args.image_path = 'PRNet/TestImages/frame2.jpg'
					args.ref_path = 'PRNet/TestImages/frame.jpg'
					texture_editing(prn, args)

					res2 = cv2.imread('PRNet/TestImages/output.jpg')
					res1[:,avg:w] = res2[:,avg:w]
					cv2.imwrite('PRNet/TestImages/res.jpg', res1)
					img_array.append(res1)
				else:
					img_array.append(frame)
			else:
				break
		print(len(img_array))
		for i in range(len(img_array)):
			out.write(img_array[i])
		out.release()
		cap.release()







if __name__ == '__main__':
	main()