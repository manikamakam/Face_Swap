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


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

def main():
	image = cv2.imread('../kang0.jpg')
	print(image.shape)
	img = deepcopy(image)
	print(img.shape)

	scale_percent = 60 # percent of original size
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

	# cv2.imshow("image", img)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()

	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	print(gray.shape)

	cv2.imshow("gray", gray)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()


	detector = dlib.get_frontal_face_detector()
	print(detector)
	predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
	print(predictor)

	rects = detector(gray, 1)
	print(rects)

	for (i, rect) in enumerate(rects):
		cv2.rectangle(resized, (rect.left(), rect.top()),  (rect.right(), rect.bottom()), (0, 255, 0), 2)

		shape = predictor(gray, rect)
		print(shape)
		shape = shape_to_np(shape)
		print(len(shape))

		for (x, y) in shape:
			cv2.circle(resized, (x, y), 1, (0, 0, 255), 2)

		# (x, y, w, h) = rect_to_bb(rect)
		# print("x", x)
		# print("y", y)
		# print("w", w)
		# print("h", h)
		

		cv2.imshow("face features", resized)
		if cv2.waitKey(0) & 0xff == 27:
			cv2.destroyAllWindows()

		# # show the face number
		# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# # loop over the (x, y)-coordinates for the facial landmarks
		# # and draw them on the image
		# for (x, y) in shape:
		# 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# show the output image with the face detections + facial landmarks















if __name__ == '__main__':
	main()