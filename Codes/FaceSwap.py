import cv2
import numpy as np
import dlib
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy import interpolate
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
		# cv2.rectangle(image, (rect.left(), rect.top()),  (rect.right(), rect.bottom()), (0, 255, 0), 2)
		# print(rect.left())
		# print(rect.right())
		# print(rect.top())
		# print(rect.bottom())
		shape = predictor(image, rect)
		# print(shape)
		shape = shape_to_np(shape)
		# print(shape)
		# print(min(shape[:,1]))
		r = (min(shape[:,0]), min(shape[:,1]), max(shape[:,0]), max(shape[:,1]))
		cv2.rectangle(img, (r[0], r[1]),  (r[2], r[3]), (0, 255, 0), 2)
		# print(r)
		# subdiv = cv2.Subdiv2D((0,0,img.shape[1],img.shape[0]))
		# c=0
		for (x, y) in shape:
			# if(c==10):
			# 	break
			cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
			# subdiv.insert((x,y))
			# c=c+1

		# triangleList = subdiv.getTriangleList();
		# print(triangleList)
		# size = img.shape
		# triangleList = triangleList.tolist()
		triangles = []
		# c=0
		# indices = []
		
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
			# c=c+1
			triangles.append(temp)
				
			# if(c==10):
			# 	break				
			
	# print(triangles)		
	print("updated",len(triangles))
	
	return img, triangles, r



def triangulation_dst(img):
	image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
	rects = detector(image, 1)

	for (i, rect) in enumerate(rects):
		# cv2.rectangle(image, (rect.left(), rect.top()),  (rect.right(), rect.bottom()), (0, 255, 0), 2)
		# print(rect.left())
		# print(rect.right())
		# print(rect.top())
		# print(rect.bottom())
		shape = predictor(image, rect)
		# print(shape)
		shape = shape_to_np(shape)
		# print(shape)
		# print(min(shape[:,1]))
		r = (min(shape[:,0]), min(shape[:,1]), max(shape[:,0]), max(shape[:,1]))
		cv2.rectangle(img, (r[0], r[1]),  (r[2], r[3]), (0, 255, 0), 2)
		# print(r)
		subdiv = cv2.Subdiv2D((0,0,img.shape[1],img.shape[0]))
		# c=0
		for (x, y) in shape:
			# if(c==10):
			# 	break
			cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
			subdiv.insert((x,y))

			# c=c+1

		triangleList = subdiv.getTriangleList();
		# print(triangleList)
		# size = img.shape
		# triangleList = triangleList.tolist()
		triangles = []
		# c=0
		indices = []
		
		for t in triangleList:
			# print(t)
			if rect_contains(r, (t[0], t[1])) and rect_contains(r, (t[2], t[3])) and rect_contains(r, (t[4], t[5])):
				# print(t) 
				temp =[]
				pt1 = (t[0],t[1])
				pt2 = (t[2],t[3])
				pt3 = (t[4],t[5])
				for i in range(len(shape)):
					if pt1==(shape[i][0],shape[i][1]):
						# print("point 1",i)
						temp.append(i)
				for j in range(len(shape)):
					if pt2 ==(shape[j][0],shape[j][1]):
						temp.append(j)
						# print("point 2",j)
				for k in range(len(shape)): 
					if pt3 == (shape[k][0],shape[k][1]):
						# print("point 3",k)
						temp.append(k)
				# print(temp)
				indices.append(temp)
				cv2.line(img, pt1, pt2, (255, 255, 255), 1)
				cv2.line(img, pt2, pt3, (255, 255, 255), 1)
				cv2.line(img, pt3, pt1, (255, 255, 255), 1)
				# c=c+1
				triangles.append(t)
				# else:
				# 	# print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
				# 	print(t)
				# 	triangleList.remove(t)
				# if(c==10):
				# 	break				
			
			# print(indices)
	# print(indices)	
	# print(triangles)		
	print("updated",len(triangles))
	
	return img, triangles, r, indices, shape

def main():

	dst_image = cv2.imread('../m2.jpeg')
	# print("dst shape",dst_image.shape)

	scale_percent = 60 # percent of original size
	width = int(dst_image.shape[1] * scale_percent / 100)
	height = int(dst_image.shape[0] * scale_percent / 100)
	dim = (width, height)
	
	dst_image = cv2.resize(dst_image, dim, interpolation = cv2.INTER_AREA)         #resized
	dst = deepcopy(dst_image)
	# print(dst.shape)
	dst, tri_dst, box_dst,indices,dst_pts =  triangulation_dst(dst)                       #triangulation
	dst_copy = deepcopy(dst_image)                                                                 
	# print(dst_res.shape)
	# print(tri_dst[0])
	# print(indices)
	src_image = cv2.imread('../ja.jpg')
	scale_percent2 = 60 # percent of original size
	width2 = int(src_image.shape[1] * scale_percent2 / 100)
	height2 = int(src_image.shape[0] * scale_percent2 / 100)
	dim2 = (width2, height2)

	src_image = cv2.resize(src_image, dim2, interpolation = cv2.INTER_AREA)
	src = deepcopy(src_image)
	# print(src.shape)
	src, tri_src, box_src =  triangulation_src(src,indices)
	# src_res = deepcopy(src_image)
	# print(tri_src[0])
	# print(box_dst)
	# print(box_src)
	hullPoints = cv2.convexHull(dst_pts, returnPoints = True)
	mask = np.zeros_like(dst_image)
	cv2.fillConvexPoly(mask, hullPoints, (255,255,255))
	mask_b = mask[:, :, 0]
	for t in range(len(tri_dst)) :
		# if rect_contains(box_dst, (tri_dst[t][0], tri_dst[t][1])) and rect_contains(box_dst, (tri_dst[t][2], tri_dst[t][3])) and rect_contains(box_dst, (tri_dst[t][4], tri_dst[t][5])) and rect_contains(box_src, (tri_src[t][0], tri_src[t][1])) and rect_contains(box_src, (tri_src[t][2], tri_src[t][3])) and rect_contains(box_src, (tri_src[t][4], tri_src[t][5])):
		# print("triangles ",tri_dst[t])
		pt1_dst = (tri_dst[t][0], tri_dst[t][1])
		pt2_dst = (tri_dst[t][2], tri_dst[t][3])
		pt3_dst = (tri_dst[t][4], tri_dst[t][5])
		# cv2.circle(dst_res, pt3_dst, 10, (0, 0, 255), 2)	
		# cv2.imshow("Destination", dst_res)
		# if cv2.waitKey(0) & 0xff == 27:
		# 	cv2.destroyAllWindows()

		# print(pt1_dst)
		pt1_src = (tri_src[t][0], tri_src[t][1])
		pt2_src = (tri_src[t][2], tri_src[t][3])
		pt3_src = (tri_src[t][4], tri_src[t][5])
		# print(pt1_src)
		bary_dst = np.linalg.inv([[pt1_dst[0], pt2_dst[0], pt3_dst[0]], [pt1_dst[1], pt2_dst[1], pt3_dst[1]], [1,1,1]])
		# print(bary_dst)
		bary_src = [[pt1_src[0], pt2_src[0], pt3_src[0]], [pt1_src[1], pt2_src[1], pt3_src[1]], [1,1,1]]

		# Bounding box of the triangle
		xleft = min(pt1_dst[0], pt2_dst[0], pt3_dst[0])
		xright = max(pt1_dst[0], pt2_dst[0], pt3_dst[0])
		ytop = min(pt1_dst[1], pt2_dst[1], pt3_dst[1])
		ybottom = max(pt1_dst[1], pt2_dst[1], pt3_dst[1])
		x_src = []
		y_src = []
		x_dst=[]
		y_dst=[]
		# if(xleft<bounding_dst[0]):
		# 	xleft=bounding_dst[0]
		# if(xright>bounding_dst[2]):
		# 	xright=bounding_dst[2]
		# if(ytop<bounding_dst[1]):
		# 	ytop=bounding_dst[1]
		# if(ybottom>bounding_dst[3]):
		# 	ybottom =bounding_dst[3]
		# print(ytop)
		# print(ybottom)
		# print(xleft)
		# print(xright)

		for x in range(int(xleft), int(xright)):
			for y in range(int(ytop), int(ybottom)):
				p = np.array([[x], [y], [1]])
				# print("destination point", p)
				bary_coords = np.dot(bary_dst, p)
				# print(bary_coords)
				alpha = bary_coords[0]
				beta = bary_coords[1]
				gamma = bary_coords[2]
				if alpha<=1 and beta<=1 and gamma<=1 and alpha>=0 and beta>=0 and gamma>=0: #and alpha+beta+gamma<=1 and alpha+beta+gamma>0:
					# print("insideeeeeeeeeeeeee")
					point = np.dot(bary_src, bary_coords)
					# print(point[0])
					x_dst.append(x)
					y_dst.append(y)
					x_src.append(point[0][0]/point[2][0])
					y_src.append(point[1][0]/point[2][0])
					# print("src_point", points_src)
		# print(len(x_src))
		# print(len(y_src))
		# print(x_dst)
		# print(y_dst)
		x_values = np.linspace(0,src_image.shape[1], src_image.shape[1],endpoint = False)
		# print(x_values)
		y_values = np.linspace(0,src_image.shape[0],src_image.shape[0], endpoint = False)
		# nx, ny = src_image.shape[1], src_image.shape[0]
		# x_values, y_values = np.meshgrid(np.arange(box_src[0], box_src[2], 1), np.arange(box_src[1], box_src[3], 1))
		# crop_src = src_image[box_src[0]:box_src[2], box_src[1]: box_src[3]]
		
		b = src_image[:,:,0]
		g = src_image[:,:,1]
		r = src_image[:,:,2]
		# print(b)
		blue = interpolate.interp2d(x_values, y_values, b, kind='cubic')
		green = interpolate.interp2d(x_values, y_values, g, kind='cubic')
		red = interpolate.interp2d(x_values, y_values, r, kind='cubic')
		# bnew= blue(x_src,y_src)
		# print(blue)
		# gne=[]
		# rnew =[]
		# print(blue(x_src[0], y_src[0]))
		# print(y_dst)
		for i in range(len(x_src)):
			bnew = blue(x_src[i], y_src[i]) 
			gnew= green(x_src[i], y_src[i]) 
			rnew = red(x_src[i], y_src[i])
			dst_copy[y_dst[i],x_dst[i]] = (bnew,gnew,rnew)


		br = cv2.boundingRect(mask_b)
		center = ((br[0] + int(br[2] / 2), br[1] + int(br[3] / 2)))
		output = cv2.seamlessClone(dst_copy, dst_image, mask_b, center, cv2.NORMAL_CLONE)





			# grid = np.mgrid[xleft:xright, ytop:ybottom].reshape(2,-1)
			# grid = np.vstack((grid, np.ones((1, grid.shape[1]))))

			# barycoords = np.dot(barytransform, grid)
			# barycoords = barycoords[:,np.all(barycoords>=0, axis=0)]
	cv2.imshow("dst_image", dst_image)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

	cv2.imshow("dst_copy", dst_copy)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()
	cv2.imshow("dst", dst)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

	cv2.imshow("output", output)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

	cv2.imshow("src_image", src_image)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()


	cv2.imshow("Source triangles", src)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()








if __name__ == '__main__':
	main()