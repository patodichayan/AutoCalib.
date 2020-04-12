import numpy as np
import glob
from helpingfunctions import*



def main():

	images = (glob.glob('Calibration_Imgs/*.jpg'))

	#Calculate Homographies and Finding Corners.
	Homographies, corner_points = corners(images)

	#Calcualte Intrinsic Camera Matrix.
	K = K_matrix(Homographies)
	prettyprint("Initial Intrinsic Camera Matrix is:",K)

	#Optimizing all the Parameters.
	K_opt,k1,k2 = optimize(K,corner_points,Homographies)
	prettyprint("Updated Intrinsic Camera Matrix is:",K_opt)
	prettyprint("Distortion Coefficents:",[k1,k2])

	#Undistort Images and Saving Rectified Images.
	undistorted_corners = UndistortImages(images,K_opt,k1,k2)	

	#Calculate Mean Reprojection Error.
	err_D, reprojected_points = ReprojectionError(K_opt,corner_points,Homographies,k1,k2)
	prettyprint("Mean Re-projection error is :",err_D)


	#Reading Rectified Images and Displaying Corners.
	imagesD = glob.glob(('Rectified/*.png'))
	imagesD.sort(key=lambda f: int(filter(str.isdigit, f)))	

	Display(imagesD,undistorted_corners,reprojected_points)
	prettyprint("","false")


if __name__ == '__main__':
    main()
