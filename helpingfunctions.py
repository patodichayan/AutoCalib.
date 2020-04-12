import numpy as np
import glob
import cv2
import copy
import math
import os
from scipy.optimize import least_squares


def showImage(image,title):
		cv2.namedWindow(title,cv2.WINDOW_NORMAL)
		cv2.resizeWindow(title,1200,1800)
		cv2.imshow(title,image)
		cv2.waitKey()
		cv2.destroyAllWindows()


def corners(images):

	#This was printed on an A4 paper and the size of each square was 
	#21.5mm. Note that the Y axis has odd number of squares (7) and X axis
	#has even number of squares (10).

	size = 21.5

	#InnerChessboardCorners.

	Nx = 9
	Ny = 6

	world_xy = np.array([[21.5, 21.5],[21.5*9,21.5],[21.5*9, 21.5*6],[21.5,21.5*6]], dtype='float32')
	# print("",world_xy)
	
	Homographies = np.zeros([3,3])
	corner_points = []

	for i in images:
		img_original = cv2.imread(i)
		img_gray = cv2.imread(i,0)

		ret, corners = cv2.findChessboardCorners(img_gray,(Nx,Ny),None)
		corner_points.append(corners)

		img_ = cv2.drawChessboardCorners(copy.deepcopy(img_original),(Nx,Ny),corners,ret)
		# showImage(img_,"drawncorners")

		#Corners wrt Original Position.

		corners_ =  np.array([[corners[0][0]],[corners[8][0]],[corners[53][0]],[corners[45][0]]],dtype='float32')

		# cv2.circle(img_original,(corners[0][0][0],corners[0][0][1]),20,[255,0,255],-1)
		# cv2.circle(img_original,(corners[8][0][0],corners[8][0][1]),20,[255,0,255],-1)
		# cv2.circle(img_original,(corners[53][0][0],corners[53][0][1]),20,[255,0,255],-1)
		# cv2.circle(img_original,(corners[45][0][0],corners[45][0][1]),20,[255,0,255],-1)
		# showImage(img_original,"Check")

		# Compute Homography.

		H,_ = cv2.findHomography(world_xy,corners_)
		Homographies = np.dstack([Homographies,H])

	return (Homographies[:,:,1:]), np.array(corner_points)


def v(H, i, j):
	return np.array([H[0, i] * H[0, j],H[0, i] * H[1, j] + 
			H[1, i] * H[0, j],H[1, i] * H[1, j],H[2, i] * H[0, j] + 
			H[0, i] * H[2, j],H[2, i] * H[1, j] + 
			H[1, i] * H[2, j],H[2, i] * H[2, j]])


def K_matrix(H):

	V_zeros = np.zeros([1,6])
	for i in range(H.shape[2]):

		#Equation 8 , Page 6 of Zhang's Paper.

		V = np.vstack([v(H[:,:,i],0,1).T,
                      (v(H[:,:,i],0,0)-v(H[:,:,i],1,1)).T])
		V_zeros = np.concatenate([V_zeros,V],axis=0)


	# V_zeros should be of shape 2n * 6. In our case n = 13.
	# Removing the row with all Zeros.

	V_zeros = V_zeros[1:]
	
	# Checking the Dimensions.
	# print(V_zeros.shape)

	#Solving for Equation 9:

	u,s,vh = np.linalg.svd(V_zeros)

	#From the linalg.svd documentation: Vector(s) with the singular values, within each vector sorted in descending order.

	b = vh[:][-1] 

	#Refer Equation 99 to 105 of "http://staff.fh-hagenberg.at/burger/publications/reports/2016Calibration/Burger-CameraCalibration-20160516.pdf".

	w = b[0]*b[2]*b[5] - (b[1]**2)*b[5] - b[0]*(b[4]**2) + 2*b[1]*b[3]*b[4] - b[2]*(b[3]**2)
	d = b[0]*b[2] - b[1]**2

	alpha = math.sqrt(w/(d*b[0]))
	beta = math.sqrt((w/(d**2))*b[0])
	gamma = math.sqrt(w/((d**2)*b[0])) * b[1] * -1
	u_c = (b[1]*b[4] - b[2]*b[3]) / d
	v_c = (b[1]*b[3] - b[0]*b[4]) / d

	return np.array([[alpha, gamma, u_c],
					[0, beta, v_c],
					[0, 0, 1]])


def calculate_lambda(K_inv,H):
	#Averaging is for precision points.
	return ((np.linalg.norm(np.matmul(K_inv,H[:,0]))+(np.linalg.norm(np.matmul(K_inv,H[:,1]))))/2)


def extrinsic(K,H):
	extrinsic_ = []
	K_inv = np.linalg.inv(K)

	for i in range(H.shape[2]):

		lambda_ = calculate_lambda(K_inv,H[:,:,i])
		A = np.dot(K_inv,H[:,:,i])/lambda_
		r1 = A[:,0]
		r2 = A[:,1]
		t = A[:,2]
		r3 = np.cross(r1,r2)
		R = np.asarray([r1, r2, r3])
		R = R.T
		extrinsic = np.zeros((3, 4))
		extrinsic[:,:-1] = R
		extrinsic[:, -1] = t
		extrinsic_.append(extrinsic)

	return extrinsic_


def intialise_param(K):
	return (np.array([K[0,0],K[1,1],K[0,2],K[1,2],K[0,1],0,0]))


def optimize(K,corner_points,Homographies):
	params = intialise_param(K)
	extrinsic_params = extrinsic(K,Homographies)
	updated_params = least_squares(fun=func,x0 = params,method="lm",args=[corner_points,extrinsic_params])
	[alpha, beta, u_c, v_c , gamma, k1, k2] = updated_params.x

	return np.array([[alpha, gamma, u_c],
					[0, beta, v_c],
					[0, 0, 1]]) , k1, k2


def World_xy(Nx,Ny):
	size = 21.5

	World_xy = []
	for i in range(1,Ny+1):
		for j in range(1,Nx+1):
			World_xy.append([size*j,size*i,0,1])
	
	return np.array(World_xy)


def func(params,corner_points,extrinsic_):
	K = np.zeros([3,3])
	K[0,0] = params[0]
	K[1,1] = params[1]
	K[0,2] = params[2]
	K[1,2] = params[3]
	K[0,1] = params[4]
	K[2,2] = 1	
	k1 = params[5]
	k2 = params[6]
	u_0 = params[2]
	v_0 = params[3]


	Nx = 9
	Ny = 6
	World_xy_ = World_xy(Nx,Ny)
	error = []
	count = 0

	for im_pts,E in zip(corner_points,extrinsic_):
		for img_points,xy_ in zip(im_pts,World_xy_):

			projected_ = np.matmul(E,xy_)
			projected_=projected_/projected_[-1]
			x_,y_ = projected_[0],projected_[1]

			U = np.matmul(K,projected_)
			U = U/U[-1]
			u_,v_ = U[0],U[1]

			#Refer Section 3.3 and Equation's 11 to 13 from Zhang's Paper.

			T = x_**2 + y_**2
			u_hat = u_ + (u_-u_0)*(k1*T + k2*(T**2))
			v_hat = v_ + (v_-v_0)*(k1*T + k2*(T**2))

			error.append(img_points[0,0] - u_hat)
			error.append(img_points[0,1] - v_hat)

	return np.float64(error).flatten()


def prettyprint(name,value):
	print("")
	if value is "false":
		pass
	else:
		print(name)
		print(value)
	

def ReprojectionError(K,corner_points,Homographies,k1,k2):
	extrinsic_params = extrinsic(K,Homographies)
	err = []
	reprojected_points = []
	u_0 = K[0,2]
	v_0 = K[1,2]

	World_ = World_xy(9,6)

	for im_pts,E in zip(corner_points,extrinsic_params):
		for img_pts,world_points in zip(im_pts,World_):

			world_x_y = np.array([[world_points[0]],[world_points[1]],[0],[1]])
			projected_points = np.matmul(E,world_x_y)
			projected_points = projected_points/projected_points[-1]
			x_,y_ = projected_points[0], projected_points[1]

			U = np.matmul(K, projected_points)
			U = U/U[-1]
			u_, v_ = U[0] , U[1]

			#Refer Section 3.3 and Equation's 11 to 13 from Zhang's Paper.

			T = x_**2 + y_**2
			u_hat = u_ + (u_-u_0)*(k1*T + k2*(T**2))
			v_hat = v_ + (v_-v_0)*(k1*T + k2*(T**2))

			reprojected_points.append([np.float32(u_hat),np.float32(v_hat)])
			err.append(math.sqrt((img_pts[0,0] - u_hat)**2 + (img_pts[0,1] - v_hat)**2))			

	reprojected_points = np.reshape(reprojected_points,(13,54,1,2))
	
	return np.mean(err) , reprojected_points


def UndistortImages(images,K,k1,k2):
	distortion = np.array([k1,k2,0,0,0],dtype=float)
	corner_points = []
	count = 0
	for i in images:
		count+=1
		img_original = cv2.imread(i) 
		img_original = cv2.undistort(img_original,K,distortion)
		# cv2.imwrite("Rectified/Rectified{}.png".format(count),img_original)
		img_gray = cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(img_gray,(9,6),None)
		corner_points.append(corners)
		img_ = cv2.drawChessboardCorners(copy.deepcopy(img_original),(9,6),corners,ret)
		# showImage(img_,"drawncorners")

	return corner_points


def Display(images,undistorted_points,reprojected_points):
	count = 0
	for i , im_points,un_points in zip(images,reprojected_points,undistorted_points):
		count+=1
		img_original = cv2.imread(i)
		img_copy = copy.deepcopy(img_original)
		for img_points,und_points in zip(im_points,un_points):

			cv2.circle(img_copy,(img_points[0][0],img_points[0][1]),20,[255,0,255])
			cv2.circle(img_copy,(und_points[0][0],und_points[0][1]),20,[255,255,0])

		# showImage(img_copy,"Difference in Corners.")
		# cv2.imwrite("Difference/Difiference{}.png".format(count),img_copy)




























	



















