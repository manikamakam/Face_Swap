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

	if video == 0 and method == 3:
		parser = argparse.ArgumentParser(description='Texture Editing by PRN')

		parser.add_argument('-i', '--image_path', default='TestImages/AFLW2000/image00081.jpg', type=str,
		                    help='path to input image')
		parser.add_argument('-r', '--ref_path', default='TestImages/trump.jpg', type=str, 
		                    help='path to reference image(texture ref)')
		parser.add_argument('-o', '--output_path', default='TestImages/output.jpg', type=str, 
		                    help='path to save output')
		parser.add_argument('--mode', default=1, type=int, 
		                    help='ways to edit texture. 0 for modifying parts, 1 for changing whole')
		parser.add_argument('--gpu', default='0', type=str, 
                        help='set gpu id, -1 for CPU')

	    # ---- init PRN
	    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu # GPU number, -1 for CPU
	    prn = PRN(is_dlib = True) 


		texture_editing(prn, parser.parse_args())





if __name__ == '__main__':
	main()