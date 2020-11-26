import numpy as np
from scipy.ndimage.filters import gaussian_filter, convolve
import matplotlib.pyplot as plt
from scipy import where
import cv2
import itertools


def get_neighbors(i, j, h, w, hw, direction):
    neighbors = []
    if direction == 0:
        neighbors = [[i, jloc] for jloc in range(j-hw, j+hw)]
    elif direction == 1:
        neighbors = [[iloc, j+iloc-i] for iloc in range(i-hw, i+hw)]
    elif direction == 2:
        neighbors = [[iloc, j] for iloc in range(i - hw, i + hw)]
    elif direction == 3:
        neighbors = [[iloc, j - iloc + i] for iloc in range(i - hw, i + hw)]
    neighbors_c = neighbors.copy()
    for t in neighbors_c:
        if t[0] not in range(0, w) or t[1] not in range(0, h):
            neighbors.remove(t)
    return neighbors

### Read Image ###
im = cv2.imread('zurlim.png', 0).astype('float')

####### Gaussian Smooth Image #######
# TODO: Implement Gaussian Smooting
# Useful functions: gaussian_filter
blurred_im = gaussian_filter(im, sigma=2.0)

###### Gradients x and y (Sobel filters) ######
# TODO: Implement Gradient along x and y
# Useful functions: convolve

im_x = convolve(blurred_im, np.array([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]]))
im_y = convolve(blurred_im, np.array([[-1, -2, -1],
                                      [0, 0, 0],
                                      [1, 2, 1]]))

###### gradient and direction ########
# TODO: Implement Gradient Magnitude and Direction

gradient = np.power((np.power(im_x, 2) + np.power(im_y, 2)), 0.5)
theta = np.arctan(im_y / im_x)
theta[np.isnan(theta)] = 0.0

thresh = 30
# TODO: Implement Thresholding criteria
thresholdEdges = blurred_im
thresholdEdges[gradient >= thresh] = 255
thresholdEdges[gradient < thresh] = 0

# TODO: Implement Non-Maximum Suppression
###### Convert to degree ######
theta = theta / np.pi * 180
###### Quantize angles ######
theta_c = theta.copy()
theta[theta_c < -45] = 0
theta[(theta_c < 0) & (theta_c >= -45)] = 1
theta[(theta_c < 45) & (theta_c >= 0)] = 2
theta[(theta_c <= 90) & (theta_c >= 45)] = 3
###### Non-maximum suppression ########
halfwidth = 2
w, h = im.shape[:2]
for i in range(w):
    for j in range(h):
        n_list = get_neighbors(i, j, h, w, hw=halfwidth, direction=theta[i][j])
        neighbors = gradient[tuple(np.asarray(n_list).T.tolist())]
        if gradient[i][j] != np.max(neighbors):
            thresholdEdges[i][j] = 0
edges = thresholdEdges

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
