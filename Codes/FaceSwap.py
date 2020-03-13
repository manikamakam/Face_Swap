import cv2
import numpy as np
import dlib
from copy import deepcopy

# Opens the Video file
# cap= cv2.VideoCapture('../data1.avi')
# i=0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     cv2.imwrite('kang'+str(i)+'.jpg',frame)
#     i+=1
 
# cap.release()
# cv2.destroyAllWindows()

# def rect_contains(rect, point) :
# 	if point[0] < rect[0] :
# 	    return False

# 	elif point[1] < rect[1] :
# 	    return False

# 	elif point[0] > rect[2] :
# 	    return False

# 	elif point[1] > rect[3] :
# 	    return False

# 	return True


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

def face(image):

	img = deepcopy(image)
	

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print("gray shape",gray.shape)

	# cv2.imshow("gray", gray)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	detector = dlib.get_frontal_face_detector()
	print(detector)
	predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
	print(predictor)

	rects = detector(gray, 1)
	print(rects)



	for (i, rect) in enumerate(rects):
		# cv2.rectangle(resized, (rect.left(), rect.top()),  (rect.right(), rect.bottom()), (0, 255, 0), 2)

		shape = predictor(gray, rect)
		print(shape)
		shape = shape_to_np(shape)
		print(len(shape))

		subdiv = cv2.Subdiv2D((0,0,gray.shape[1],gray.shape[0]))
		for (x, y) in shape:
			cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
			subdiv.insert((x,y))

		draw_delaunay(img, subdiv)

	return img


def main():

	dst = cv2.imread('../kang0.jpg')
	print("dst shape",dst.shape)

	scale_percent = 60 # percent of original size
	width = int(dst.shape[1] * scale_percent / 100)
	height = int(dst.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	dst = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
	print("dst resized shape", dst.shape)
	dst_res = face(dst)

	src = cv2.imread('../ja.jpg')
	print("src shape",src.shape)

	src = cv2.resize(src, dim, interpolation = cv2.INTER_AREA)
	print("src resized shape", src.shape)
	src_res = face(src)



	cv2.imshow("Destination", dst)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

	cv2.imshow("Source", src)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()


	cv2.imshow("Destination Image", dst_res)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

	cv2.imshow("Source Image", src_res)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()




# show the output image with the face detections + facial landmarks















if __name__ == '__main__':
	main()