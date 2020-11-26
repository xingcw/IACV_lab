import numpy as np
from scipy.ndimage.filters import gaussian_filter,convolve
import matplotlib.pyplot as plt
from scipy import where
import cv2

### Read Image ###
im=cv2.imread('/home/cvcourse/pics/zurlim.png', 0).astype('float')


####### Gaussian Smooth Image #######
# TODO: Implement Gaussian Smooting
# Useful functions: gaussian_filter
#blurred_im = ...


###### Gradients x and y (Sobel filters) ######
# TODO: Implement Gradient along x and y
# Useful functions: convolve

#im_x = ... 
#im_y = ...


###### gradient and direction ########
# TODO: Implement Gradient Magnitude and Direction

#gradient = ...
#theta = ...


thresh=30;
# TODO: Implement Thresholding criteria
#thresholdEdges = ...


# TODO: Implement Non-Maximum Suppression
###### Convert to degree ######
#theta = ...
###### Quantize angles ######
#
###### Non-maximum suppression ########
### ...
### ...
#edges= ...


# Plotting of results
# No need to change it
plt.close("all")
plt.ion()
f, ax_arr = plt.subplots(1, 2, figsize=(18, 16))
ax_arr[0].set_title("Input Image")
ax_arr[1].set_title("Canny Edge Detector")
ax_arr[0].imshow(im, cmap='gray')
ax_arr[1].imshow(edges, cmap='gray')
plt.show()
plt.pause(5)

