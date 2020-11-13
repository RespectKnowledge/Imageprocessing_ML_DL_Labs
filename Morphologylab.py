# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:13:18 2020

@author: Abdul Qayyum
"""
#%%
# source https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
#Morphological Transformations
#Goal
#In this Lesson
#We will learn different morphological operations like Erosion, Dilation, Opening, Closing etc.
#We will see different functions like : cv2.erode(), cv2.dilate(), cv2.morphologyEx() etc.

#Theory
#Morphological transformations are some simple operations based on the image shape. 
#It is normally performed on binary images. It needs two inputs, one is our original image, 
#second one is called structuring element or kernel which decides the nature of operation. 
#Two basic morphological operators are Erosion and Dilation. Then its variant forms 
#like Opening, Closing, Gradient etc also comes into play. 
#We will see them one-by-one with help of following image:

#1. Erosion
#The basic idea of erosion is just like soil erosion only, 
#it erodes away the boundaries of foreground object (Always try to keep foreground in white). 
#So what does it do? The kernel slides through the image (as in 2D convolution). 
#A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, 
#otherwise it is eroded (made to zero).
#
#So what happends is that, all the pixels near boundary will be discarded depending upon the size of kernel. 
#So the thickness or size of the foreground object decreases or simply white region decreases in the image. 
#It is useful for removing small white noises (as we have seen in colorspace chapter), detach two connected objects etc.
#
#Here, as an example, I would use a 5x5 kernel with full of ones. Let’s see it how it works:
import cv2
import numpy as np

img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

#2. Dilation
#It is just opposite of erosion. Here, a pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’. 
#So it increases the white region in the image or size of foreground object increases.
# Normally, in cases like noise removal, erosion is followed by dilation. 
# Because, erosion removes white noises, but it also shrinks our object. 
# So we dilate it. Since noise is gone, they won’t come back, but our object area increases. 
# It is also useful in joining broken parts of an object.

dilation = cv2.dilate(img,kernel,iterations = 1)

#3. Opening
#Opening is just another name of erosion followed by dilation. 
#It is useful in removing noise, as we explained above. 
#Here we use the function, cv2.morphologyEx()

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#
#4. Closing
#Closing is reverse of Opening, Dilation followed by Erosion. 
#It is useful in closing small holes inside the foreground objects, 
#or small black points on the object.
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#5. Morphological Gradient
#It is the difference between dilation and erosion of an image.
#
#The result will look like the outline of the object.

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

#6. Top Hat
#It is the difference between input image and Opening of the image. 
#Below example is done for a 9x9 kernel.
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

#7. Black Hat
#It is the difference between the closing of the input image and input image.

blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

#Structuring Element
#We manually created a structuring elements in the previous examples with help of Numpy. 
#It is rectangular shape. But in some cases, you may need elliptical/circular shaped kernels. 
#So for this purpose, OpenCV has a function, cv2.getStructuringElement(). 
#You just pass the shape and size of the kernel, you get the desired kernel.



# Rectangular Kernel
cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# Elliptical Kernel
cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


# Cross-shaped Kernel
cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

#%% cv2 images 
#https://www.geeksforgeeks.org/difference-between-opening-and-closing-in-digital-image-processing/?ref=rp
# Python program to transform an image using 
# threshold. 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
  
# Image operation using thresholding 
#img = cv2.imread('c4.jpg') 
  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
img
ret, thresh = cv2.threshold(img, 0, 255, 
                            cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU) 
cv2.imshow('image', thresh)


# Noise removal using Morphological 
# closing operation 
kernel = np.ones((3, 3), np.uint8) 
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                            kernel, iterations = 2) 
  
# Background area using Dialation 
bg = cv2.dilate(closing, kernel, iterations = 1) 
  
# Finding foreground area 
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
ret, fg = cv2.threshold(dist_transform, 0.02
                        * dist_transform.max(), 255, 0) 
  
cv2.imshow('image', fg) 
#%% https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html
#Morphological Filtering
#Morphological image processing is a collection of non-linear operations related 
#to the shape or morphology of features in an image, such as boundaries, skeletons, etc. 
#In any given technique, we probe an image with a small shape or template called a structuring element, 
#which defines the region of interest or neighborhood around a pixel.
#
#In this document we outline the following basic morphological operations:
#
#Erosion
#
#Dilation
#
#Opening
#
#Closing
#
#White Tophat
#
#Black Tophat
#
#Skeletonize
#
#Convex Hull
#
#To get started, let’s load an image using io.imread. Note that morphology functions 
#only work on gray-scale or binary images, so we set as_gray=True.

import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_ubyte
from skimage import io

orig_phantom = img_as_ubyte(data.shepp_logan_phantom())
fig, ax = plt.subplots()
ax.imshow(orig_phantom, cmap=plt.cm.gray)

#Let’s also define a convenience function for plotting comparisons:
def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    
#1. Erosion
#Morphological erosion sets a pixel at (i, j) to the minimum over all pixels 
#in the neighborhood centered at (i, j). 
#The structuring element, selem, passed to erosion is a boolean array that describes this neighborhood. 
#Below, we use disk to create a circular structuring element, which we use for most of the following examples.
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

selem = disk(6)
eroded = erosion(orig_phantom, selem)
plot_comparison(orig_phantom, eroded, 'erosion')

#Notice how the white boundary of the image disappears or gets eroded as we
#increase the size of the disk. Also notice the increase in size of 
#the two black ellipses in the center and the disappearance of the 3 light grey patches in the lower part of the image.

#Dilation
#Morphological dilation sets a pixel at (i, j) to the maximum over all pixels in 
#the neighborhood centered at (i, j). Dilation enlarges bright regions and shrinks dark regions.
dilated = dilation(orig_phantom, selem)
plot_comparison(orig_phantom, dilated, 'dilation')

#Opening
#Morphological opening on an image is defined as an erosion followed by a dilation. 
#Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks.

opened = opening(orig_phantom, selem)
plot_comparison(orig_phantom, opened, 'opening')

#Closing
#Morphological closing on an image is defined as a dilation followed by an erosion. 
#Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks.
#
#To illustrate this more clearly, let’s add a small crack to the white border:

phantom = orig_phantom.copy()
phantom[10:30, 200:210] = 0

closed = closing(phantom, selem)
plot_comparison(phantom, closed, 'closing')




