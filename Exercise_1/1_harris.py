from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.signal import convolve2d
import numpy as np
import cv2
import itertools


# Implement a function that performs non-maximum suppression. You can leave it for the end
def get_neighbors(i, j, h, w, hw):
    neighbors = list(itertools.product(range(i - hw, i + hw), range(j - hw, j + hw)))
    neighbors_c = neighbors.copy()
    for t in neighbors_c:
        if t[0] not in range(0, w) or t[1] not in range(0, h):
            neighbors.remove(t)
    x = np.unique(np.asarray([t[0] for t in neighbors]))
    y = np.unique(np.asarray([t[1] for t in neighbors]))
    neighbors = [x.tolist(), y.tolist()]
    return neighbors


def nonmax_suppression(harris_resp, thr, halfwidth=2):
    # Outputs:
    # 1) cornersy: list with row coordinates of identified corner pixels.
    # 2) cornersx: list with respective column coordinates of identified corner pixels.
    # Elements from the two lists with the same index must correspond to the same corner.

    cornersy = []
    cornersx = []
    h, w = im.shape[:2]

    # TODO: perform non-maximum suppression

    for i in range(w):
        for j in range(h):
            n_list = get_neighbors(i, j, h, w, hw=halfwidth)
            neighbors = harris_resp[np.ix_(n_list[0], n_list[1])]
            if harris_resp[i][j] >= thr and harris_resp[i][j] == np.max(neighbors):
                cornersx.append(i)
                cornersy.append(j)

    return cornersy, cornersx


# Implement the main part of the exercise

# Define parameters
sigma_w = 2.0
sigma_d = 2.0
kappa = 0.04
rot_angle = 60
thresh = 800

# Read the image
im = cv2.imread('test.png', 0)
im = im.astype('float')

# Rotation of the image
if rot_angle != 0:
    im = rotate(im, rot_angle)

# TODO: Implement Harris corners
# Useful functions: gaussian_filter1d, gaussian_filter

# im_x = gaussian_filter1d(im, axis=0, sigma=sigma_d, order=1)
# im_y = gaussian_filter1d(im, axis=1, sigma=sigma_d, order=1)

I_x = convolve2d(im, np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]]))
I_y = convolve2d(im, np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]]))
I_x2_bar = gaussian_filter(np.multiply(I_x, I_x), sigma=sigma_w)
I_y2_bar = gaussian_filter(np.multiply(I_y, I_y), sigma=sigma_w)
I_xy_bar = gaussian_filter(np.multiply(I_x, I_y), sigma=sigma_w)
H = np.zeros(shape=(im.shape[0], im.shape[1]))
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        A = np.asarray([[I_x2_bar[i][j], I_xy_bar[i][j]], [I_xy_bar[i][j], I_y2_bar[i][j]]])
        H[i][j] = np.linalg.det(A) - kappa * (np.multiply(np.trace(A), np.trace(A)))
corn = nonmax_suppression(H, thresh, 5)

# Visualization of the results

# Plotting of results
# No need to change it
plt.close("all")
# plt.ion()
f, ax_arr = plt.subplots(1, 3, figsize=(18, 16))
ax_arr[0].set_title("Input Image")
ax_arr[1].set_title("Harris Response")
ax_arr[2].set_title("Detections")
ax_arr[0].imshow(im, cmap='gray')
ax_arr[1].imshow(H, cmap='gray')
ax_arr[2].imshow(im, cmap='gray')
ax_arr[2].scatter(x=corn[0], y=corn[1], c='r', s=10)
plt.show()
# plt.pause(5)
