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

	if len(rects) >= 2:
		shape = []
		for (i, rect) in enumerate(rects):
			s = predictor(gray, rect)
			s = shape_to_np(s)
			shape.append(s)

	else:
		for (i, rect) in enumerate(rects):
			shape = predictor(gray, rect)
			shape = shape_to_np(shape)

			# Uncomment the following to view facial features
			# for (x, y) in shape:
			# 	cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
	
	shape = np.asarray(shape)
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
		for pt in dst_pts:
			mes.append(pt[0])
			mes.append(pt[1])

		mes = np.asarray(mes, np.float32)
		mes = np.resize(mes, (136,1))
		# print(mes.shape)

		if initialized_once == False:
			initialized_once = True
			state[0:136] = mes
			state[136:272] = np.reshape(np.asarray([0] * 136), (136,1))
			kalman.statePost = state

		else:
			kalman.correct(mes)

	else:
		kalman.statePost = state
		dst_pts = np.asarray([[0] * 2] * 68)

		if initialized_once == False:
			return dst, True, dst_flag, False
			# dst_pts = np.asarray([[0] * 2] * 68)

	for i in range(len(dst_pts)):
		dst_pts[i][0] = state[2*i]
		dst_pts[i][1] = state[2*i+1]	

	dst_copy = deepcopy(dst)

	# print(dst_pts)

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

	return output, True, dst_flag, initialized_once

def selectRects(rects1, rects2):
	min1 = 10000
	for (x, y) in rects1:
		if x < min1:
			min1 = x

	min2 = 10000
	for (x, y) in rects2:
		if x < min2:
			min2 = x

	if min1 < min2:
		return rects1, rects2

	else:
		return rects2, rects1		


def faceSwapVideo(img, kalman, found, initialized_once):
	rects, flag = face(img)
	# if (flag == False):
	# 	return None, False

	rnd = deepcopy(img)
	src = deepcopy(img)
	dst = deepcopy(img)
	dst_copy = deepcopy(img)	

	# kalman predict
	state = kalman.predict()	

	if flag and len(rects) >= 2:
		src_pts, dst_pts = selectRects(rects[0], rects[1])
		# src_pts = rects[0]
		# dst_pts = rects[1]

		# for (x, y) in src_pts:
		# 	cv2.circle(rnd, (x, y), 1, (0, 0, 255), 2)

		# for (x, y) in dst_pts:
		# 	cv2.circle(rnd, (x, y), 1, (255, 0, 0), 2)

		# cv2.imshow("Rxd", rnd)
		# cv2.waitKey(10)

		mes = []
		if flag:
			for pt in src_pts:
				mes.append(pt[0])
				mes.append(pt[1])
			for pt in dst_pts:
				mes.append(pt[0])
				mes.append(pt[1])

			mes = np.asarray(mes, np.float32)
			mes = np.resize(mes, (272,1))
			# print(mes.shape)

			if initialized_once == False:
				initialized_once = True
				state[0:272] = mes
				state[272:544] = np.reshape(np.asarray([0] * 272), (272,1))
				kalman.statePost = state

			else:
				kalman.correct(mes)

	else:
		print("Cannot detect both faces")
		kalman.statePost = state
		src_pts = np.asarray([[0] * 2] * 68)
		dst_pts = np.asarray([[0] * 2] * 68)

		if initialized_once == False:
			return img, True, flag, False

	for i in range(len(src_pts)):
		src_pts[i][0] = state[2*i]
		src_pts[i][1] = state[2*i+1]

	for i in range(len(dst_pts)):
		dst_pts[i][0] = state[2*i + 136]
		dst_pts[i][1] = state[2*i+1 + 136]

	# swap first face
	# generate mask for first swap
	hullPoints1 = cv2.convexHull(dst_pts, returnPoints = True)
	m1 = np.zeros_like(img)
	cv2.fillConvexPoly(m1, hullPoints1, (255,255,255))
	mask1 = m1[:, :, 0]

	# generate P matrix
	P1 = []
	for (x, y) in dst_pts:
		P1.append([x, y, 1])
	P1 = np.asarray(P1)

	# generate X Y matrix
	X1 = np.zeros(len(P1)+3)
	Y1 = np.zeros(len(P1)+3)
	for i in range(len(src_pts)):
		X1[i] = src_pts[i][0]
		Y1[i] = src_pts[i][1]

	# parameters for the thin plate spines
	x_const1, y_const1 = getTPScoff(P1, X1, Y1)

	# swap faces
	output1 = swap(x_const1, y_const1, P1, src, dst, dst_copy, mask1)

	# cv2.imshow("Output", output1)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	# cv2.imshow("dst_copy", dst_copy)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	dst1 = deepcopy(output1)


	# swap second face
	temp = src_pts
	src_pts = dst_pts
	dst_pts = temp

	# generate mask for first swap
	hullPoints2= cv2.convexHull(dst_pts, returnPoints = True)
	m2 = np.zeros_like(img)
	cv2.fillConvexPoly(m2, hullPoints2, (255,255,255))
	mask2 = m2[:, :, 0]

	# generate P matrix
	P2 = []
	for (x, y) in dst_pts:
		P2.append([x, y, 1])
	P2 = np.asarray(P2)

	# generate X Y matrix
	X2 = np.zeros(len(P2)+3)
	Y2 = np.zeros(len(P2)+3)
	for i in range(len(src_pts)):
		X2[i] = src_pts[i][0]
		Y2[i] = src_pts[i][1]

	# parameters for the thin plate spines
	x_const2, y_const2 = getTPScoff(P2, X2, Y2)

	# swap faces
	output = swap(x_const2, y_const2, P2, src, dst1, output1, mask2)	

	# cv2.imshow("Output", output)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	# cv2.imshow("dst_copy", output1)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	return output, True, flag, initialized_once

	# else:
	# 	print("Cannot detect both faces") 
	# 	return img, False, flag, initialized_once


def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--video')

	Args = Parser.parse_args()
	video = int(Args.video)
	img_array = []

	if video == 0:
		pass
		cap = cv2.VideoCapture('../TestSet_P2/Test2.mp4')
		if (cap.isOpened() == False):
			print("Unable to read camera feed")
		frame_width = int(cap.get(3))
		frame_height = int(cap.get(4))
		# scale_percent = 100  # percent of original size
		# frame_width = int(frame_width * scale_percent / 100)
		# frame_height = int(frame_height * scale_percent / 100)
		out = cv2.VideoWriter('Test2OutputTSP.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width, frame_height))
		count = 1

		# kalman filter
		kalman = cv2.KalmanFilter(544, 272)

		# generate transition matrix
		t_mat = np.asarray([[0] * 544] * 544, np.float32)
		for i in range(t_mat.shape[0]):
			# for j in range(mes_mat.shape[1]):
				# if i == j or j == i + 272:
			t_mat[i][i] = 1.0
			if i+272 < 544:
				t_mat[i][i+272] = 1.0

		# generate measurement matrix
		m_mat = np.asarray([[0] * 544] * 272, np.float32)
		for i in range(m_mat.shape[0]):
			m_mat[i][i] = 1.0


		kalman.measurementMatrix = m_mat
		kalman.transitionMatrix = t_mat

		# Process Covariance Matrix (Q)
		kalman.processNoiseCov = np.identity(544, np.float32) * 1e-5
		# Measurement Noise Covariance Matrix (R)
		kalman.measurementNoiseCov = np.identity(272, np.float32) * 1e-2
		kalman.errorCovPost = np.identity(544, np.float32)
		found = False
		initialized_once = False

		while (True):
			ret, frame = cap.read()
			if ret == True:
				# dst = frame
				print(count)
				# print(src.shape)
				# print(dst.shape)
				
				# Swap faceswap
				# if count > 6:
				# output, succ, found, initialized_once = faceSwap(src, dst, kalman, found, initialized_once)
				output, succ, found, initialized_once = faceSwapVideo(frame, kalman, found, initialized_once)
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
				# if count == 50:
				# 	break
			else:
				break

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
		out = cv2.VideoWriter('Test1OutputTSP.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width, frame_height))

		count = 1

		# kalman filter
		kalman = cv2.KalmanFilter(272, 136)

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
				output, succ, found, initialized_once = faceSwap(src, dst, kalman, found, initialized_once)
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