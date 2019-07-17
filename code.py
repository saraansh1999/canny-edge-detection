import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

# {200, 500, 1000, 1500: (100, 255, 0.1, 0.3), 2000: (100, 255, 0.2, 0.2), 2240: (100, 255, 0.55, 0.5)}

weak = 100
strong = 255
lowThresholdRatio = 0.2
highThresholdRatio = 0.2

def gaussian_kernel(size, sigma=1):
	size = int(size) // 2
	x, y = np.mgrid[-size:size+1, -size:size+1]
	normal = 1 / (2.0 * np.pi * sigma**2)
	g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
	return g

def sobel_filters(img):
	Kx = np.array([ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ], np.float32)
	Ky = np.array([ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ], np.float32)

	Ix = ndimage.filters.convolve(img, Kx)
	Iy = ndimage.filters.convolve(img, Ky)


	G = np.hypot(Ix, Iy)
	G = G / G.max() * 255
	theta = np.arctan2(Iy, Ix)

	return (G, theta)

def non_max_suppression(img, D):
	M, N = img.shape
	Z = np.zeros((M, N), dtype=np.int32)
	angle = D * 180.0 / np.pi
	angle[angle < 0] += 180

	for i in range(1, M-1):
		for j in range(1, N-1):
			try:
				q = 255
				r = 255

				#0
				if(0 <= angle[i, j] < 22.5 or 157.5 <= angle[i,j] <= 180):
					q = img[i, j+1]
					r = img[i, j-1]
				#45
				elif(22.5 <= angle[i, j] <= 67.5):
					q = img[i+1, j-1]
					r = img[i-1, j+1]
				#90
				elif(67.5 <= angle[i, j] < 112.5):
					q = img[i+1, j]
					r = img[i-1, j]
				#135
				elif(112.5 <= angle[i, j] <= 157.5):
					q = img[i-1, j-1]
					r = img[i+1, j+1]

				if(img[i, j] >= q and img[i, j] >= r):
					Z[i, j] = img[i, j]
				else:
					Z[i, j] = 0

			except IndexError as e:
				pass

	return Z

def threshold(img):

	highThreshold = img.max() * highThresholdRatio
	lowThreshold = highThreshold * lowThresholdRatio

	M, N = img.shape
	res = np.zeros((M, N), dtype=np.int32)

	strong_i, strong_j = np.where(img >= highThreshold)
	zeros_i, zeros_j = np.where(img < lowThreshold)

	weak_i, weak_j = np.where((img <= highThreshold) & (img > lowThreshold))

	res[strong_i, strong_j] = strong
	res[weak_i, weak_j] = weak

	return res

def hysteresis(img):
	
	M, N = img.shape

	for i in range(1, M-1):
		for j in range(1, N-1):
			if(img[i, j] == weak):
				try:
					if( (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong) ):
						img[i, j] = strong
					else:
						img[i, j] = 0
				except IndexError as e:
					pass

	return img

def detect(img):

	img_smoothed = ndimage.filters.convolve(img, gaussian_kernel(3, 1))
	gradients, thetas = sobel_filters(img_smoothed)
	nonMaxImg = non_max_suppression(gradients, thetas)
	thresholdImg = threshold(nonMaxImg)
	img_final = hysteresis(thresholdImg)

	return img_final

#-----Reading the image-----------------------------------------------------

img_original = cv2.imread('2000m.png', 1)

#-----Converting image to LAB Color model----------------------------------- 
# lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# cv2.imshow("lab",lab)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----Splitting the LAB image to different channels-------------------------
# l, a, b = cv2.split(lab)
# cv2.imshow('l_channel', l)
# cv2.imshow('a_channel', a)
# cv2.imshow('b_channel', b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----Applying CLAHE to L-channel-------------------------------------------
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl = clahe.apply(l)
# cv2.imshow('CLAHE output', cl)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
# limg = cv2.merge((cl,a,b))
# cv2.imshow('limg', limg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----Converting image from LAB Color model to RGB model--------------------
# final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# cv2.imshow('final', final)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#_____END_____#

img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img = img.astype(np.float32)
img = detect(img)
plt.subplot(1, 2, 1), plt.imshow(img_original, cmap='gray')
plt.subplot(1, 2, 2), plt.imshow(img, cmap='gray')
plt.show()