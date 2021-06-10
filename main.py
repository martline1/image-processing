#!/usr/bin/python

import cv2
import numpy as np
from skimage import io, img_as_float

def main():
	img = img_as_float(io.imread("Lola.jpeg", as_gray = True))

	# Gauss
	img_gauss = cv2.GaussianBlur(img, (5, 5), 5, borderType = cv2.BORDER_CONSTANT)

	# Laplace
	laplace = cv2.Laplacian(img_gauss, cv2.CV_64F, ksize = 5)

	# Canny necesita la imagen con el filtro laplace y dos thresholds para X y Y
	canny = cv2.Canny(np.uint8(laplace), 100, 200)

	cv2.imshow("Original", img)
	cv2.imshow("Gauss", img_gauss)
	cv2.imshow("Laplace", laplace)
	cv2.imshow("Canny", canny)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
