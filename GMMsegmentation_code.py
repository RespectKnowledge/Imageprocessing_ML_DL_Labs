# -*- coding: utf-8 -*-
"""
Created on Fri May 29 08:31:49 2020

@author: Abdul Qayyum
"""


import numpy as np
import cv2


#path='D:\\LAB4Segmentation\\LAB6\\12.png'
path='D:\\LAB4Segmentation\\LAB6\\train_image_patient2_15.png'
#img = cv2.imread("images/BSE_Image.jpg")
img = cv2.imread(path)

# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1,3))  #-1 reshape means, in this case MxN

from sklearn.mixture import GaussianMixture as GMM

#covariance choices, full, tied, diag, spherical
gmm_model = GMM(n_components=4, covariance_type='tied').fit(img2)  #tied works better than full
gmm_labels = gmm_model.predict(img2)

#Put numbers back to original shape so we can reconstruct segmented image
original_shape = img.shape
segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
cv2.imwrite("segmented1.jpg", segmented*(255/4))