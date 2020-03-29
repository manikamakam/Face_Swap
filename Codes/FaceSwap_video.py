import cv2
import numpy as np
import dlib
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy import interpolate



def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
def triangulation_src(img,indices):
	image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
	rects = detector(image, 1)

	for (i, rect) in enumerate(rects):
		shape = predictor(image, rect)
		shape = shape_to_np(shape)
		r = (min(shape[:,0]), min(shape[:,1]), max(shape[:,0]), max(shape[:,1]))
		cv2.rectangle(img, (r[0], r[1]),  (r[2], r[3]), (0, 255, 0), 2)
		for (x, y) in shape:
			cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
		triangles = []

		for t in indices:
			temp =[]
			pt1 = (shape[t[0]][0],shape[t[0]][1])
			pt2 = (shape[t[1]][0],shape[t[1]][1])
			pt3 = (shape[t[2]][0],shape[t[2]][1])
			temp.append(shape[t[0]][0])
			temp.append(shape[t[0]][1])
			temp.append(shape[t[1]][0])
			temp.append(shape[t[1]][1])
			temp.append(shape[t[2]][0])
			temp.append(shape[t[2]][1])

			cv2.line(img, pt1, pt2, (255, 255, 255), 1)
			cv2.line(img, pt2, pt3, (255, 255, 255), 1)
			cv2.line(img, pt3, pt1, (255, 255, 255), 1)
			triangles.append(temp)				
	# print("updated",len(triangles))
	# cv2.imshow("src_image", img)
	# if cv2.waitKey(0) & 0xff == 27:
	# 	cv2.destroyAllWindows()
	return img, triangles, r



def triangulation_dst(img,rects,predictor):
	for (i, rect) in enumerate(rects):
		shape = predictor(img, rect)
		shape = shape_to_np(shape)
		r = (min(shape[:,0]), min(shape[:,1]), max(shape[:,0]), max(shape[:,1]))
		cv2.rectangle(img, (r[0], r[1]),  (r[2], r[3]), (0, 255, 0), 2)
		subdiv = cv2.Subdiv2D((0,0,img.shape[1],img.shape[0]))

		for (x, y) in shape:
			cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
			subdiv.insert((x,y))

		triangleList = subdiv.getTriangleList();
		triangles = []
		indices = []
		
		for t in triangleList:
			if rect_contains(r, (t[0], t[1])) and rect_contains(r, (t[2], t[3])) and rect_contains(r, (t[4], t[5])):
				temp =[]
				pt1 = (t[0],t[1])
				pt2 = (t[2],t[3])
				pt3 = (t[4],t[5])
				for i in range(len(shape)):
					if pt1==(shape[i][0],shape[i][1]):
						temp.append(i)
				for j in range(len(shape)):
					if pt2 ==(shape[j][0],shape[j][1]):
						temp.append(j)
				for k in range(len(shape)): 
					if pt3 == (shape[k][0],shape[k][1]):
						temp.append(k)
				indices.append(temp)
				cv2.line(img, pt1, pt2, (255, 255, 255), 1)
				cv2.line(img, pt2, pt3, (255, 255, 255), 1)
				cv2.line(img, pt3, pt1, (255, 255, 255), 1)
				
				triangles.append(t)	

	# print("updated",len(triangles))
	
	return img, triangles, r, indices, shape

def main():
	img_array = []
	cap = cv2.VideoCapture('../TestSet_P2/Test1.mp4')
	if (cap.isOpened() == False):
		print("Unable to read camera feed")
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	scale_percent = 60  # percent of original size
	frame_width = int(frame_width * scale_percent / 100)
	frame_height = int(frame_height * scale_percent / 100)
	out = cv2.VideoWriter('Test1OutputTri.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width, frame_height))


	while (True):
		ret, frame = cap.read()
		c=0 
		if ret == True:
			c= c+1
			scale_percent = 60 # percent of original size
			width = int(frame.shape[1] * scale_percent / 100)
			height = int(frame.shape[0] * scale_percent / 100)
			dim = (width, height)
			
			dst_image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)         #resized
			dst = deepcopy(dst_image)

			dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
			detector = dlib.get_frontal_face_detector()
			predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
			rects = detector(dst, 1)
			if len(rects) > 0:	
				
				dst, tri_dst, box_dst,indices,dst_pts =  triangulation_dst(dst,rects,predictor) 
				
				dst_copy = deepcopy(dst_image)                                                                 

				src_image = cv2.imread('../TestSet_P2/Rambo.jpg')
				scale_percent2 = 60 # percent of original size
				width2 = int(src_image.shape[1] * scale_percent2 / 100)
				height2 = int(src_image.shape[0] * scale_percent2 / 100)
				dim2 = (width2, height2)

				src_image = cv2.resize(src_image, dim2, interpolation = cv2.INTER_AREA)
				src = deepcopy(src_image)
				src, tri_src, box_src =  triangulation_src(src,indices)

				hullPoints = cv2.convexHull(dst_pts, returnPoints = True)
				mask = np.zeros_like(dst_image)
				cv2.fillConvexPoly(mask, hullPoints, (255,255,255))
				mask_b = mask[:, :, 0]

				x_src = []
				y_src = []
				x_dst=[]
				y_dst=[]
				for t in range(len(tri_dst)) :
					pt1_dst = (tri_dst[t][0], tri_dst[t][1])
					pt2_dst = (tri_dst[t][2], tri_dst[t][3])
					pt3_dst = (tri_dst[t][4], tri_dst[t][5])

					pt1_src = (tri_src[t][0], tri_src[t][1])
					pt2_src = (tri_src[t][2], tri_src[t][3])
					pt3_src = (tri_src[t][4], tri_src[t][5])

					bary_dst = np.linalg.inv([[pt1_dst[0], pt2_dst[0], pt3_dst[0]], [pt1_dst[1], pt2_dst[1], pt3_dst[1]], [1,1,1]])

					bary_src = [[pt1_src[0], pt2_src[0], pt3_src[0]], [pt1_src[1], pt2_src[1], pt3_src[1]], [1,1,1]]

					xleft = min(pt1_dst[0], pt2_dst[0], pt3_dst[0])
					xright = max(pt1_dst[0], pt2_dst[0], pt3_dst[0])
					ytop = min(pt1_dst[1], pt2_dst[1], pt3_dst[1])
					ybottom = max(pt1_dst[1], pt2_dst[1], pt3_dst[1])


					for x in range(xleft, xright):
						for y in range(ytop, ybottom):
							p = np.array([[x], [y], [1]])
							
							bary_coords = np.dot(bary_dst, p)
							alpha = bary_coords[0]
							beta = bary_coords[1]
							gamma = bary_coords[2]
							if alpha<=1 and beta<=1 and gamma<=1 and alpha>=0 and beta>=0 and gamma>=0: #and alpha+beta+gamma<=1 and alpha+beta+gamma>0:
								
								point = np.dot(bary_src, bary_coords)
								
								x_dst.append(x)
								y_dst.append(y)
								x_src.append(point[0][0]/point[2][0])
								y_src.append(point[1][0]/point[2][0])
				# print(len(x_src))
				# print(len(y_src))
				# print(len(x_dst))
				# print(len(y_dst))
				x_values = np.linspace(0,src_image.shape[1], src_image.shape[1],endpoint = False)

				y_values = np.linspace(0,src_image.shape[0],src_image.shape[0], endpoint = False)

				b = src_image[:,:,0]
				g = src_image[:,:,1]
				r = src_image[:,:,2]

				blue = interpolate.interp2d(x_values, y_values, b, kind='cubic')
				green = interpolate.interp2d(x_values, y_values, g, kind='cubic')
				red = interpolate.interp2d(x_values, y_values, r, kind='cubic')

				for i in range(len(x_src)):
					bnew = blue(x_src[i], y_src[i]) 
					gnew= green(x_src[i], y_src[i]) 
					rnew = red(x_src[i], y_src[i])
					dst_copy[y_dst[i],x_dst[i]] = (bnew,gnew,rnew)


				br = cv2.boundingRect(mask_b)
				center = ((br[0] + int(br[2] / 2), br[1] + int(br[3] / 2)))
				output = cv2.seamlessClone(dst_copy, dst_image, mask_b, center, cv2.NORMAL_CLONE)
				# cv2.imshow("output", dst_copy)
				# if cv2.waitKey(0) & 0xff == 27:
				# 	cv2.destroyAllWindows()
				img_array.append(output)

			# if c==1:
			# 	cv2.imshow("output", output)
			# 	if cv2.waitKey(0) & 0xff == 27:
			# 		cv2.destroyAllWindows()
			# 	break

		else:
			break
	print(len(img_array))		
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()
	cap.release()

		# cv2.imshow("dst_image", dst_image)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

		# cv2.imshow("dst_copy", dst_copy)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()
		# cv2.imshow("dst", dst)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

		# cv2.imshow("output", output)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

		# cv2.imshow("src_image", src_image)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

		# cv2.imshow("Source triangles", src)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()








if __name__ == '__main__':
	main()