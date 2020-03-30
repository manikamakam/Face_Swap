import cv2
import numpy as np
import dlib
from copy import deepcopy
import math
from scipy import interpolate
import argparse


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

def face(image):
	img = deepcopy(image)

	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# get facial features using dlib
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
	rects = detector(gray, 1)

	if (len(rects) == 0):
		print("no face found")
		return None, False

	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)

		# Uncomment the following to view facial features

		for (x, y) in shape:
			cv2.circle(img, (x, y), 1, (0, 0, 255), 2)

			# cv2.imshow("Face", img)
			# if cv2.waitKey(0) & 0xff == 27:
			# 	cv2.destroyAllWindows()
	
	# cv2.imshow("Face", img)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	return shape, True

def getTPScoff(P, X, Y):
	temp = np.zeros((len(P)+3, len(P)+3))

	for i in range(len(P)):
		for j in range(len(P)):
			r = math.sqrt((P[i][0] - P[j][0])**2 + (P[i][1] - P[j][1])**2)
			temp[i][j] = r**2 * math.log(r**2 + 1e-6)

	for i in range(len(P)):
		for j in range(len(P), len(P)+3):
			temp[i][j] = P[i][j-len(P)]
			temp[j][i] = P[i][j-len(P)]

	I = np.identity(len(P) + 3)

	lam = 1e-6
	# lam = 0.00000000001

	temp2 = temp + lam*I
	x_const = np.matmul(np.linalg.inv(temp2), X)
	y_const = np.matmul(np.linalg.inv(temp2), Y)
	
	return x_const, y_const


def swap(x_const, y_const, P, src, dst, dst_copy, mask):

	random = np.zeros_like(dst)
	for i in range(dst.shape[0]):
		for j in range(dst.shape[1]):
			if (mask[i][j] == 255):
				random[i][j] = dst[i][j]

	cv2.imshow("Random", random)
	cv2.waitKey(10)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	src_copy = deepcopy(src)

	b = src[:,:,0]
	g = src[:,:,1]
	r = src[:,:,2]

	blue = interpolate.interp2d(range(src.shape[1]), range(src.shape[0]), b, kind='cubic')
	green = interpolate.interp2d(range(src.shape[1]), range(src.shape[0]), g, kind='cubic')
	red = interpolate.interp2d(range(src.shape[1]), range(src.shape[0]), r, kind='cubic')
	
	# for i in range(len(P)):
	# 	l = len(x_const)
	# 	s_x = 0
	# 	s_y = 0
	# 	for j in range(len(P)):
	# 		r = math.sqrt((P[j][0] - P[i][0])**2 + (P[j][1] - P[i][1])**2)
	# 		u = r**2 * math.log(r**2 + 1e-6)
	# 		s_x = s_x + x_const[j]*u
	# 		s_y = s_y + y_const[j]*u

	# 	# print(s_x)
	# 	# print(s_y)

	# 	x_new = x_const[l-1] + x_const[l-3]*P[i][0] + x_const[l-2]*P[i][1] + s_x
	# 	y_new = y_const[l-1] + y_const[l-3]*P[i][0] + y_const[l-2]*P[i][1] + s_y

	# 	# print(x_new, y_new, dst_pts[i][0], dst_pts[i][1])

	# 	cv2.circle(src_copy, (int(x_new), int(y_new)), 1, (255, 0, 0), 2)

	flag = False
	
	for pt in P:
		if pt[0] != 0 or pt[1] != 0:
			flag = True

	if not flag:
		return dst	

	for xi in range(dst.shape[1]):
		for yi in range(dst.shape[0]):
			if (mask[yi][xi] == 255):
				l = len(x_const)
				s_x = 0
				s_y = 0
				for j in range(len(P)):
					r = math.sqrt((P[j][0] - xi)**2 + (P[j][1] - yi)**2)
					u = r**2 * math.log(r**2 + 1e-6)
					s_x = s_x + x_const[j]*u
					s_y = s_y + y_const[j]*u

				# (x, y) coordinates of src image
				x_new = x_const[l-1] + x_const[l-3]*xi + x_const[l-2]*yi + s_x
				y_new = y_const[l-1] + y_const[l-3]*xi + y_const[l-2]*yi + s_y

				# print(x_new, y_new)

				# replace pixels in dst_copy with interpolated values
				dst_copy[yi][xi][0] = blue(x_new, y_new) 
	 			dst_copy[yi][xi][1] = green(x_new, y_new) 
	 			dst_copy[yi][xi][2] = red(x_new, y_new)

	r = cv2.boundingRect(mask)
	center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
	output = cv2.seamlessClone(dst_copy, dst, mask, center, cv2.NORMAL_CLONE)

	return output
	# return src_copy

def faceSwap(src, dst, kalman, found, initialized_once):
	if found:
		# kalman predict
		state = kalman.predict()		

	# kalman predict
	state = kalman.predict()
	# print(state)

	# er = 0
	# for i in range(len(dst_pts)):
	# 	er = er + abs(dst_pts[i][0] - state[2*i])
	# 	er = er + abs(dst_pts[i][1] - state[2*i+1])

	# print(er/len(dst_pts))

	dst_pts, dst_flag = face(dst)
	# if (dst_flag == False):
	# 	return None, False
	src_pts, src_flag = face(src)
	if (src_flag == False):
		print("No face found in source?? What is happening?? ")
		return None, False

	mes = []
	if dst_flag:
		if initialized_once == False:
			initialized_once = True
		
		for pt in dst_pts:
			mes.append(pt[0])
			mes.append(pt[1])

		mes = np.asarray(mes, np.float32)
		mes = np.resize(mes, (136,1))
		# print(mes.shape)

		kalman.correct(mes)

	else:
		kalman.statePost = state

		if initialized_once == False:
			dst_pts = np.asarray([[0] * 2] * 68)

	for i in range(len(dst_pts)):
		dst_pts[i][0] = state[2*i]
		dst_pts[i][1] = state[2*i+1]	

	dst_copy = deepcopy(dst)

	# generate mask
	hullPoints = cv2.convexHull(dst_pts, returnPoints = True)
	m = np.zeros_like(dst)
	cv2.fillConvexPoly(m, hullPoints, (255,255,255))
	mask = m[:, :, 0]

	# generate P matrix
	P = []
	for (x, y) in dst_pts:
		P.append([x, y, 1])
	P = np.asarray(P)

	# generate X Y matrix
	X = np.zeros(len(P)+3)
	Y = np.zeros(len(P)+3)
	for i in range(len(src_pts)):
		X[i] = src_pts[i][0]
		Y[i] = src_pts[i][1]

	# parameters for the thin plate spines
	x_const, y_const = getTPScoff(P, X, Y)

	# swap faces
	output = swap(x_const, y_const, P, src, dst, dst_copy, mask)	

	return output, True, dst_flag

def faceSwapVideo(img):
	print("dst")
	dst_pts, dst_flag = face(dst)
	if (dst_flag == False):
		return None, False
	print("src")
	src_pts, src_flag = face(src)
	if (dst_flag == False or src_flag == False):
		return None, False

	dst_copy = deepcopy(dst)

	# generate mask
	hullPoints = cv2.convexHull(dst_pts, returnPoints = True)
	m = np.zeros_like(dst)
	cv2.fillConvexPoly(m, hullPoints, (255,255,255))
	mask = m[:, :, 0]

	# generate P matrix
	P = []
	for (x, y) in dst_pts:
		P.append([x, y, 1])
	P = np.asarray(P)

	# generate X Y matrix
	X = np.zeros(len(P)+3)
	Y = np.zeros(len(P)+3)
	for i in range(len(src_pts)):
		X[i] = src_pts[i][0]
		Y[i] = src_pts[i][1]

	# parameters for the thin plate spines
	x_const, y_const = getTPScoff(P, X, Y)

	# swap faces
	output = swap(x_const, y_const, P, src, dst, dst_copy, mask)	

	return output, True


# def main():

# 	# load source image
# 	src = cv2.imread('../sg1.jpg')
# 	# resize src
# 	scale_percent = 60 
# 	width1 = int(src.shape[1] * scale_percent / 100)
# 	height1 = int(src.shape[0] * scale_percent / 100)
# 	dim1 = (width1, height1)
# 	src = cv2.resize(src, dim1, interpolation = cv2.INTER_AREA)

# 	# load destination image
# 	dst = cv2.imread('../a2.jpg')
# 	# resize dst
# 	scale_percent = 60 # percent of original size
# 	width2 = int(dst.shape[1] * scale_percent / 100)
# 	height2 = int(dst.shape[0] * scale_percent / 100)
# 	dim2 = (width2, height2)
# 	dst = cv2.resize(dst, dim2, interpolation = cv2.INTER_AREA)

# 	# Swap faceswap
# 	output = faceSwap(src, dst)

# 	cv2.imshow("Destination", output)
# 	if cv2.waitKey(0) & 0xff == 27:
# 		cv2.destroyAllWindows()

# 	cv2.imwrite("../output.png", output)

def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--video')

	Args = Parser.parse_args()
	video = int(Args.video)
	img_array = []

	if video ==0:
		pass
		# cap = cv2.VideoCapture('../TestSet_P2/Test2.mp4')
		# if (cap.isOpened() == False):
		# 	print("Unable to read camera feed")
		# frame_width = int(cap.get(3))
		# frame_height = int(cap.get(4))
		# # scale_percent = 100  # percent of original size
		# # frame_width = int(frame_width * scale_percent / 100)
		# # frame_height = int(frame_height * scale_percent / 100)
		# out = cv2.VideoWriter('Test2OutputTri.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width, frame_height))
		# while (True):
		# 	ret, frame = cap.read()
		# 	if ret == True:
		# 		# scale_percent = 100 # percent of original size
		# 		# width = int(frame.shape[1] * scale_percent / 100)
		# 		# height = int(frame.shape[0] * scale_percent / 100)
		# 		# dim = (width, height)
				
		# 		# dst_image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)         #resized
		# 		dst = deepcopy(dst_image)

		# 		dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
		# 		detector = dlib.get_frontal_face_detector()
		# 		predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
		# 		rects = detector(dst, 1)
		# 		if len(rects) ==2:	
		# 			tri_dst, tri_src,dst_pts =  triangulation(dst,rects,predictor,1) 
		# 			tri_dst2, tri_src2,dst_pts2 =  triangulation(dst,rects,predictor,0) 

		# 			dst_copy = deepcopy(dst_image)    

		# 			hullPoints = cv2.convexHull(dst_pts, returnPoints = True)
		# 			mask = np.zeros_like(dst_image)
		# 			cv2.fillConvexPoly(mask, hullPoints, (255,255,255))
		# 			mask_b = mask[:, :, 0]

		# 			x_dst, y_dst, x_src, y_src= get_points(tri_dst, tri_src)
		# 			x_dst2, y_dst2, x_src2, y_src2= get_points(tri_dst2,tri_src2)

		# 			b = dst_image[:,:,0]
		# 			g = dst_image[:,:,1]
		# 			r = dst_image[:,:,2]

		# 			blue = interpolate.interp2d(range(dst_image.shape[1]), range(dst_image.shape[0]), b, kind='cubic')
		# 			green = interpolate.interp2d(range(dst_image.shape[1]), range(dst_image.shape[0]), g, kind='cubic')
		# 			red = interpolate.interp2d(range(dst_image.shape[1]), range(dst_image.shape[0]), r, kind='cubic')

		# 			for i in range(len(x_src)):
		# 				bnew = blue(x_src[i], y_src[i]) 
		# 				gnew= green(x_src[i], y_src[i]) 
		# 				rnew = red(x_src[i], y_src[i])
		# 				dst_copy[y_dst[i],x_dst[i]] = (bnew,gnew,rnew)
		# 			for i in range(len(x_src2)):
		# 				bnew2 = blue(x_src2[i], y_src2[i]) 
		# 				gnew2= green(x_src2[i], y_src2[i]) 
		# 				rnew2 = red(x_src2[i], y_src2[i])
		# 				dst_copy[y_dst2[i],x_dst2[i]] = (bnew2,gnew2,rnew2)

		# 			hullPoints2 = cv2.convexHull(dst_pts2, returnPoints = True)
		# 			mask2 = np.zeros_like(dst_image)
		# 			cv2.fillConvexPoly(mask2, hullPoints2, (255,255,255))
		# 			mask_b2 = mask2[:, :, 0]
			
		# 			br = cv2.boundingRect(mask_b)
		# 			center = ((br[0] + int(br[2] / 2), br[1] + int(br[3] / 2)))
		# 			output = cv2.seamlessClone(dst_copy, dst_image, mask_b, center, cv2.NORMAL_CLONE)
				
		# 			br2 = cv2.boundingRect(mask_b2)
		# 			center2 = ((br2[0] + int(br2[2] / 2), br2[1] + int(br2[3] / 2)))
		# 			output2 = cv2.seamlessClone(dst_copy, output, mask_b2, center2, cv2.NORMAL_CLONE)
		# 			img_array.append(output2)
		# 	else:
		# 		break
	else:
		cap = cv2.VideoCapture('../TestSet_P2/Test1.mp4')
		if (cap.isOpened() == False):
			print("Unable to read camera feed")
		frame_width = int(cap.get(3))
		frame_height = int(cap.get(4))
		src = cv2.imread('../TestSet_P2/Rambo.jpg')
		# scale_percent = 100  # percent of original size
		# frame_width = int(frame_width * scale_percent / 100)
		# frame_height = int(frame_height * scale_percent / 100)
		out = cv2.VideoWriter('Test1OutputTri.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width, frame_height))

		count = 1

		# kalman filter
		kalman = cv2.KalmanFilter(272,136)

		# generate transition matrix
		t_mat = np.asarray([[0] * 272] * 272, np.float32)
		for i in range(t_mat.shape[0]):
			# for j in range(mes_mat.shape[1]):
				# if i == j or j == i + 136:
			t_mat[i][i] = 1.0
			if i+136 < 272:
				t_mat[i][i+136] = 1.0

		# generate measurement matrix
		m_mat = np.asarray([[0] * 272] * 136, np.float32)
		for i in range(m_mat.shape[0]):
			m_mat[i][i] = 1.0


		kalman.measurementMatrix = m_mat
		kalman.transitionMatrix = t_mat

		# Process Covariance Matrix (Q)
		kalman.processNoiseCov = np.identity(272, np.float32) * 1e-3
		# Measurement Noise Covariance Matrix (R)
		kalman.measurementNoiseCov = np.identity(136, np.float32) * 1e-1
		kalman.errorCovPost = np.identity(272, np.float32)
		found = False
		initialized_once = False

		while (True):
			ret, frame = cap.read()
			if ret == True:
				# load destination image
				dst = frame
				print(count)
				# print(src.shape)
				# print(dst.shape)
				
				# Swap faceswap
				# if count > 6:
				output, succ, found = faceSwap(src, dst, kalman, found, initialized_once)
				# else:
				# 	count = count + 1
				# 	continue

				if succ == True:
					# cv2.imshow("output", output)
					# cv2.waitKey(1)
					img_array.append(output)
				else:
					img_array.append(frame)
				count = count + 1
			else:
				break

	print(len(img_array))		
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()
	cap.release()



# show the output image with the face detections + facial landmarks

if __name__ == '__main__':
	main()