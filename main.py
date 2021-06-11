#!/usr/bin/python

import cv2
import numpy as np
from skimage import io, img_as_float

def gaussTransform(imgPath, show = False, title = ""):
	img = img_as_float(io.imread(imgPath, as_gray = True))

	gauss = cv2.GaussianBlur(img, (5, 5), 5, borderType = cv2.BORDER_CONSTANT)

	if show:
		cv2.imshow(title, gauss)
	else:
		return gauss

def laplaceTransform(imgPath, show = False, title = ""):
	gauss = gaussTransform(imgPath)

	laplace = cv2.Laplacian(gauss, cv2.CV_64F, ksize = 5)

	if show:
		cv2.imshow(title, laplace)
	else:
		return laplace

def cannyTransform(title, imgPath):
	laplace = laplaceTransform(imgPath)

	canny = cv2.Canny(np.uint8(laplace), 100, 200)

	cv2.imshow(title, canny)

def main():
	gaussTransform("Ajolote.jpeg", title = "Ajolote sin Ruido", show = True)
	laplaceTransform("chapala.jpeg", title = "Chapala líneas", show = True)
	laplaceTransform("Chucky.jpeg", title = "Chucky líneas", show = True)
	gaussTransform("Lola.jpeg", title = "Lola sin ruido", show = True)
	laplaceTransform("Paloma.jpeg", title = "Paloma líneas", show = True)
	gaussTransform("pez.jpeg", title = "Pez sin ruido", show = True)
	cannyTransform("Serpiente laplace y Canny", "serpiente.jpeg")

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
