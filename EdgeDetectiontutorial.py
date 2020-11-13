# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:30:33 2020

@author: Abdul Qayyum
"""

#%% Lab for edge detection
# Install anaconda  https://www.anaconda.com/distribution/
# Go run and open anaconda navigator, once anaconda nevigator is open, just install spyeder and after the installation of sypeder,
# open the spyeder IDE.
# go run and  open anaconda prompt
# install the following libraries such as numpy, matplotlib and scikit-learn etc
# pip install numpy, 
#pip install matplotlib
#pip install, 
#pip install scikit-image  https://scikit-image.org/docs/0.13.x/install.html
#python -m pip install --upgrade scikit-image
# pip install opencv-python

#Roberts
#The idea behind the Roberts cross operator is to approximate the gradient of an
#image through discrete differentiation which is achieved by computing the sum of the squares of the
#differences between diagonally adjacent pixels. It highlights regions of high spatial gradient which often
#correspond to edges.
#
#Sobel:
#Similar to Roberts - calculates gradient of the image. 
#The operator uses two 3×3 kernels which are convolved with the original image to calculate
#approximations of the derivatives – one for horizontal changes, and one for vertical.
#
#Scharr:
#Typically used to identify gradients along the x-axis (dx = 1, dy = 0) and y-axis (dx = 0,
#dy = 1) independently. Performance is quite similar to Sobel filter.
#
#Prewitt:
#The Prewitt operator is based on convolving
#the image with a small, separable, and integer valued filter in horizontal and vertical directions and is
#therefore relatively inexpensive in terms of computations like Sobel operator.
#
#Farid:
#Farid and Simoncelli propose to use a pair of kernels, one for interpolation and another for
#differentiation (csimilar to Sobel). These kernels, of fixed sizes 5 x 5 and 7 x 7, are optimized so
#that the Fourier transform approximates their correct derivative relationship. 
#
#Canny:
#The Process of Canny edge detection algorithm can be broken down to 5 different steps:
#1. Apply Gaussian filter to smooth the image in order to remove the noise
#2. Find the intensity gradients of the image
#3. Apply non-maximum suppression to get rid of spurious response to edge detection
#4. Apply double threshold to determine potential edges (supplied by the user)
#5. Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that
#are weak and not connected to strong edges.
#%% Define your image
from skimage import io, filters, feature
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import numpy as np

path='D:\\prof-Machine-learning\\python_for_microscopists-master\\python_for_microscopists-master1\\my_image\\Eiffel_Tower.jpeg'
img = cv2.imread(path, 0)
#Edge detection
from skimage.filters import roberts, sobel, scharr, prewitt

roberts_img = roberts(img)
sobel_img = sobel(img)
scharr_img = scharr(img)
prewitt_img = prewitt(img)
#farid_img = farid(img)

cv2.imshow("Roberts", roberts_img)
cv2.imshow("Sobel", sobel_img)
cv2.imshow("Scharr", scharr_img)
cv2.imshow("Prewitt", prewitt_img)
#cv2.imshow("Farid", farid_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Canny
canny_edge = cv2.Canny(img,50,80)

#Autocanny
sigma = 0.3 # you can check this value for your own prposes.
median = np.median(img)

# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - sigma) * median)) # take the sigma and median value 
upper = int(min(255, (1.0 + sigma) * median))
auto_canny = cv2.Canny(img, lower, upper)


cv2.imshow("Canny", canny_edge)
cv2.imshow("Auto Canny", auto_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ################################ canny filter using skimage###########################3
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature


##################################3Generate noisy image of a square########################
#im = np.zeros((128, 128))
#im[32:-32, 32:-32] = 1
img = img.astype(np.float64)
im = ndi.rotate(img, 15, mode='constant')
im = ndi.gaussian_filter(im, 4)
im += 0.2 * np.random.random(im.shape)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=3)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title(r'Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title(r'Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()

######################################### edges detection in sckit-image #################################
import matplotlib.pyplot as plt

from skimage import filters
from skimage.data import camera
from skimage.util import compare_images


#image = camera()
edge_roberts = filters.roberts(img)
edge_sobel = filters.sobel(img)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

#Ridge filters can be used to detect ridge-like structures, such as neurites, tubes, vessels, wrinkles.
#The present class of ridge filters relies on the eigenvalues of the Hessian matrix of image intensities to detect ridge
#structures where the intensity changes perpendicular but not along the structure.

from skimage import io, filters, feature
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

#Ridge operators 
#https://scikit-image.org/docs/dev/auto_examples/edges/plot_ridge_filter.html#sphx-glr-auto-examples-edges-plot-ridge-filter-py
from skimage.filters import meijering, sato, frangi, hessian

#img = io.imread("images/leaf.jpg")
img = rgb2gray(img)

#sharpened = unsharp_mask(image0, radius=1.0, amount=1.0)
meijering_img = meijering(img)
sato_img = sato(img)
frangi_img = frangi(img)
hessian_img = hessian(img)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(meijering_img, cmap='gray')
ax2.title.set_text('Meijering')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(sato_img, cmap='gray')
ax3.title.set_text('Sato')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(frangi_img, cmap='gray')
ax4.title.set_text('Frangi')
plt.show()