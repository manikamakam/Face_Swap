import cv2
import numpy as np
import dlib
from copy import deepcopy
import math
from scipy import interpolate


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

def draw_delaunay(img, subdiv) :
	triangleList = subdiv.getTriangleList();
	# size = img.shape

	# r = (0, 0, size[1], size[0])

	for t in triangleList :

		pt1 = (t[0], t[1])

		pt2 = (t[2], t[3])

		pt3 = (t[4], t[5])
		 

	# if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
		cv2.line(img, pt1, pt2, (255, 255, 255), 1)
		cv2.line(img, pt2, pt3, (255, 255, 255), 1)
		cv2.line(img, pt3, pt1, (255, 255, 255), 1)

def face(img):

	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# get facial features using dlib
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
	rects = detector(gray, 1)

	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)

		# Uncomment the following to view facial features

		# for (x, y) in shape:
		# 	cv2.circle(img, (x, y), 1, (0, 0, 255), 2)

	# cv2.imshow("Face", img)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	return shape 

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

	# print(X)
	# print(Y)
	lam = 0.00000000001

	temp2 = temp + lam*I
	x_const = np.matmul(np.linalg.inv(temp2), X)
	y_const = np.matmul(np.linalg.inv(temp2), Y)
	
	return x_const, y_const


def swap(x_const, y_const, P, src, dst, dst_copy, mask_b, x_min, x_max, y_min, y_max):

	random = np.zeros_like(dst)
	for i in range(dst.shape[0]):
		for j in range(dst.shape[1]):
			if (mask_b[i][j] == 255):
				random[i][j] = dst[i][j]

	# cv2.imshow("Destination", random)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	b = src[:,:,0]
	g = src[:,:,1]
	r = src[:,:,2]

	blue = interpolate.interp2d(range(src.shape[1]), range(src.shape[0]), b, kind='cubic')
	green = interpolate.interp2d(range(src.shape[1]), range(src.shape[0]), g, kind='cubic')
	red = interpolate.interp2d(range(src.shape[1]), range(src.shape[0]), r, kind='cubic')
			


	for xi in range(dst.shape[1]):
		for yi in range(dst.shape[0]):
			if (mask_b[yi][xi] == 255):
				l = len(x_const)
				s_x = 0
				s_y = 0
				for j in range(len(P)):
					r = math.sqrt((P[j][0] - xi)**2 + (P[j][1] - yi)**2)
					u = r**2 * math.log(r**2 + 1e-6)
					s_x = s_x + x_const[j]*u
					s_y = s_y + y_const[j]*u

				# print(s_x)
				# print(s_y)

				x_new = x_const[l-1] + x_const[l-3]*xi + x_const[l-2]*yi + s_x
				y_new = y_const[l-1] + y_const[l-3]*xi + y_const[l-2]*yi + s_y

				dst_copy[yi][xi][0] = blue(x_new, y_new) 
	 			dst_copy[yi][xi][1] = green(x_new, y_new) 
	 			dst_copy[yi][xi][2] = red(x_new, y_new)

				# print(x_new, y_new)

				# cv2.circle(src, (int(x_new), int(y_new)), 1, (255, 0, 0), 2)

	r = cv2.boundingRect(mask_b)
	center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
	output = cv2.seamlessClone(dst_copy, dst, mask_b, center, cv2.NORMAL_CLONE)

	return output

def faceSwap(src, dst):
	dst_pts = face(dst)

	return output


def main():

	dst = cv2.imread('../a2.jpg')
	dst_copy = deepcopy(dst)
	scale_percent = 60 # percent of original size
	width1 = int(dst.shape[1] * scale_percent / 100)
	height1 = int(dst.shape[0] * scale_percent / 100)
	dim1 = (width1, height1)
	dst = cv2.resize(dst, dim1, interpolation = cv2.INTER_AREA)
	dst_pts = face(dst)

	src = cv2.imread('../sg1.jpg')
	scale_percent = 60 # percent of original size
	width2 = int(src.shape[1] * scale_percent / 100)
	height2 = int(src.shape[0] * scale_percent / 100)
	dim2 = (width2, height2)
	# dst = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
	src = cv2.resize(src, dim2, interpolation = cv2.INTER_AREA)
	src_pts = face(src)

	hullPoints = cv2.convexHull(dst_pts, returnPoints = True)

	# print(hullIndex)

	mask = np.zeros_like(dst)
	cv2.fillConvexPoly(mask, hullPoints, (255,255,255))
	mask_b = mask[:, :, 0]
	# _, mask_b = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

	# print(mask_b.shape)

	# cv2.imshow("Mask", mask_b)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()



	P = []
	for (x, y) in dst_pts:
		# print(x, y)
		P.append([x, y, 1])

	X = np.zeros(len(P)+3)
	Y = np.zeros(len(P)+3)
	for i in range(len(src_pts)):
		X[i] = src_pts[i][0]
		Y[i] = src_pts[i][1]

	x_max = int(np.amax(X[0:-3]))
	x_min = int(np.amin(X[0:-3]))
	y_max = int(np.amax(Y[0:-3]))
	y_min = int(np.amin(Y[0:-3]))

	P = np.asarray(P)

	x_const, y_const = getTPScoff(P, X, Y)

	output = swap(x_const, y_const, P, src, dst, dst_copy, mask_b, x_min, x_max, y_min, y_max)	

	cv2.imshow("Destination", output)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

	cv2.imwrite("../output.png", output)


# show the output image with the face detections + facial landmarks

if __name__ == '__main__':
	main()