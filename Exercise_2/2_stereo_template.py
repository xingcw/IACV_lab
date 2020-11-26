from __future__ import print_function
import cv2
import numpy as np
import argparse
import time
from matplotlib import pyplot as plt
import plotly.graph_objects as go  # Plotly can be installed using the command "pip install plotly".


def main():
    """Main function for exercise 2, Stereo Vision.

    Usage:
        python stereo.py -part all
                     -left /path_to_image_from_left_camera \
                     -right /path_to_image_from_right_camera  \
                     -baseline 1
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-part", type=str, help="Part to run. Can be all, triangulation, ncc, ncc_fast, or stereo",
                        default="all")
    parser.add_argument("-left", type=str, help="input file name (intensity image from left camera)",
                        default="./tsukuba_left.pgm")
    parser.add_argument("-right", type=str, help="input file name (intensity image from right camera)",
                        default="./tsukuba_right.pgm")
    parser.add_argument("-baseline", help="distance between the two cameras", type=float, default=1)
    parser.add_argument("-focal_length", help="focal length in inch", type=float, default=1.378)
    parser.add_argument("-aperture_x", help="size of the image plane in inch (horizontal)",
                        type=float, default=1.417)
    parser.add_argument("-aperture_y", help="size of the image plane in inch (vertical)",
                        type=float, default=0.945)

    args = parser.parse_args()

    left = cv2.imread(args.left, -1)
    right = cv2.imread(args.right, -1)

    if left is None or right is None:
        raise Exception('Please make sure the image paths are correct!')

    # Prepare camera parameter dict
    camera_parameters = {'baseline': args.baseline, 'focal_length': args.focal_length,
                         'aperture_x': args.aperture_x, 'aperture_y': args.aperture_y}

    if args.part == 'all' or args.part == 'triangulation':
        # -----------------------------------------------------------------------------------------------------------
        #                                          Test triangulation
        # -----------------------------------------------------------------------------------------------------------

        m_width = 640       # width of image
        m_height = 480      # height image
        x_left = np.array([640, 640 / 2 + 1, 314])  # x-coordinates of points in the left image
        x_right = np.array([0, 640 / 2 - 1, 52])    # x-coordinates of points in the right image
        y = np.array([480 / 2, 480 / 2, 163])       # y-coordinates of points (same for both images)

        points_sol = np.array([[0.500, 0.000, 0.972], [0.500, 0.000, 311.193], [-0.023, -0.261, 2.376]])

        print("Testing triangulate()")
        points = triangulate(x_left, x_right, y, m_width, m_height, camera_parameters)

        if np.allclose(points, points_sol, rtol=1e-2):
            print("Test of triangulate() successful :)\n\n")
        else:
            print("ERROR!!! Test of triangulate() failed :(\n\n")

    if args.part == 'all' or args.part == 'ncc':
        # -----------------------------------------------------------------------------------------------------------
        #                                    Test normalized cross-correlation
        # -----------------------------------------------------------------------------------------------------------
        # Test on a dummy patch
        patch = np.array([[0, 0, 0, 0],
                          [0, 1, -1, 0],
                          [0, 0, 0, 0]])

        corr_sol = np.array([[[1.0, -0.5],
                              [-0.5, 1.0]]])

        print("Testing compute_correlation")
        corr = compute_ncc(patch, patch, 1)

        if np.allclose(corr, corr_sol, rtol=1e-2):
            print("Test of compute_correlation() successful :)\n\n")
        else:
            print("ERROR!!! Test of compute_correlation() failed :(\n\n")

        # Test on a actual image
        # Compute the NCC only on a crop to save time
        left_crop = left[140:240, 125:225]
        right_crop = right[140:240, 125:225]
        print("Computing NCC on a real image")
        t0 = time.time()
        corr = compute_ncc(left_crop, right_crop, 5)
        t1 = time.time()
        print("Computation took {:.2f} seconds. \n\n".format(t1 - t0))

        # Plot the computed NCC for a particular column in the left image
        column_index = 40
        plot_correlation(left_crop, right_crop, corr, column_index)

        # Pause for more time if needed
        plt.pause(5)
        plt.close()

    if args.part == 'all' or args.part == 'ncc_fast':
        # -----------------------------------------------------------------------------------------------------------
        #                            Test faster version of normalized cross-correlation
        # -----------------------------------------------------------------------------------------------------------

        print("Running two approaches to compute mean")
        # Initialize a random numpy array
        a = np.random.rand(1000, 1000, 5)

        # Approach 1: Naive for loop
        t0 = time.time()
        a_mean_naive = np.zeros((1000, 1000))
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a_mean_naive[i, j] = a[i, j, :].mean()

        t1 = time.time()
        print('Computation took {:.3f} seconds using for loops'.format(t1 - t0))

        # Approach 2: Numpy operations
        t0 = time.time()
        a_mean_numpy = a.mean(axis=2)
        t1 = time.time()
        print('Computation took {:.3f} seconds using numpy operation \n\n'.format(t1 - t0))

        # Test on a dummy patch
        patch = np.array([[0, 0, 0, 0],
                          [0, 1, -1, 0],
                          [0, 0, 0, 0]])

        corr_sol = np.array([[[1.0, -0.5],
                              [-0.5, 1.0]]])

        print("Testing compute_correlation_fast")
        corr_fast = compute_ncc_fast(patch, patch, 1)

        if np.allclose(corr_fast, corr_sol, rtol=1e-2):
            print("Test of compute_correlation_fast() successful :)\n\n")
        else:
            print("ERROR!!! Test of compute_correlation_fast() failed :(\n\n")

        # Test on a actual image
        left_crop = left[140:240, 125:225]
        right_crop = right[140:240, 125:225]
        print("Computing NCC on a real image")
        t0 = time.time()
        corr = compute_ncc_fast(left_crop, right_crop, 5)
        t1 = time.time()
        print("Computation took {:.2f} seconds with the fast version.\n\n".format(t1 - t0))

        # Plot if needed
        column_index = 40
        plot_correlation(left_crop, right_crop, corr, column_index)

        # Pause for more time if needed
        plt.pause(5)
        plt.close()

    if args.part == 'all' or args.part == 'stereo':
        # -----------------------------------------------------------------------------------------------------------
        #                                          Test define_points_3d
        # -----------------------------------------------------------------------------------------------------------

        # Perform the 3D reconstruction
        print("Performing 3D reconstruction")
        t0 = time.time()
        points3d = define_points_3d(left, right, 5, camera_parameters)
        t1 = time.time()
        print("Computation took {:.2f} seconds. \n".format(t1 - t0))

        # During execution of the above code, the python interpreter might generate a warning that zero divisions are occurring. Can you explain why?

        print("Plotting reconstructed point cloud. This should open in the browser.")
        plot_point_cloud(left, points3d)
        plt.pause(5)

        print("Done!")


def triangulate(x_left, x_right, y, m_width, m_height, camera_parameters):
    """Triangulate (determine 3D world coordinates) a set of points given their projected coordinates in two images.

    Args:
        x_left (np.array of shape (num_points,)): Projected x-coordinates of the 3D-points in the left image
        x_right (np.array of shape (num_points,)): Projected x-coordinates of the 3D-points in the right image
        y (np.array of shape (num_points,)): Projected y-coordinates of the 3D-points (same for both images)
        m_width (int): width of the image
        m_height (int): height of the image
        camera_parameters (dict): Dict containing camera parameters
    Returns:
        points (np.array of shape (num_points, 3): triangulated 3D co-ordinates of the input points in world
                                                   coordinates
    """

    baseline = camera_parameters['baseline']
    focal_length = camera_parameters['focal_length']
    aperture_x = camera_parameters['aperture_x']
    aperture_y = camera_parameters['aperture_y']

    points = np.zeros((x_left.shape[0], 3))

    # TODO: Perform triangulation
    #
    # ...
    #

    return points


def compute_ncc(gray_left, gray_right, mask_halfwidth):
    """Calculate normalized cross-correlation (NCC) between patches at the same row in two images. The regions
    near the boundary of the image, where the patches go out of image, are ignored. That is, for an input image,
    "mask_halfwidth" number of rows and columns will be ignored on each side.

    For input images of size (num_rows, num_cols), the output will be an array of size
    (num_rows - 2*mask_halfwidth, num_cols - 2*mask_halfwidth, num_cols - 2*mask_halfwidth). The value
    output[r, c_l, c_r] denotes the NCC between the patch centered at (r + mask_halfwidth, c_l + mask_halfwidth)
    in the left image and the patch centered at  (r + mask_halfwidth, c_r + mask_halfwidth) at the right image.

    Args:
        gray_left (np.array of shape (num_rows, num_cols)): left grayscale image
        gray_right (np.array of shape (num_rows, num_cols)): right grayscale image
        mask_halfwidth (int): Half-size of the square neighbourhood used for computing NCC. Thus a patch of size
                              (2*mask_halfwidth+1, 2*mask_halfwidth+1) will be used.

    Returns:
        corr (np.array of shape (num_rows - 2*mask_halfwidth, num_cols - 2*mask_halfwidth,
        num_cols - 2*mask_halfwidth)): Array containing the normalized cross-correlation (NCC) between patches
                                       in the two images. The value corr[r, c_l, c_r] denotes the NCC between
                                       the patch centered at (r + mask_halfwidth, c_l + mask_halfwidth) in the
                                       left image and the patch centered at
                                       (r + mask_halfwidth, c_r + mask_halfwidth) at the right image.
    """

    m_height, m_width = gray_left.shape

    corr = np.zeros((m_height - 2 * mask_halfwidth, m_width - 2 * mask_halfwidth, m_width - 2 * mask_halfwidth))

    # Loop over the rows. Ignore the boundary rows, where the patches go out of image
    for y in range(mask_halfwidth, m_height - mask_halfwidth):
        # Loop over patches in left image
        for x_l in range(mask_halfwidth, m_width - mask_halfwidth):
            # TODO extract the patch from the left image and normalize it
            #
            # ...
            #

            # Loop over patches in the right image in the same scan line, i.e. same y coordinate
            for x_r in range(mask_halfwidth, m_width - mask_halfwidth):
                # TODO extract the patch from the right image and normalize it
                #
                # ...
                #

                # TODO Compute correlation
                #
                # ...
                #
                pass

    return corr


def compute_ncc_fast(gray_left, gray_right, mask_halfwidth):
    """ Faster version of compute_ncc().
    Args:
        gray_left (np.array of shape (num_rows, num_cols)): left grayscale image
        gray_right (np.array of shape (num_rows, num_cols)): right grayscale image
        mask_halfwidth (int): Half-size of the square neighbourhood used for computing NCC. Thus a patch of size
                              (2*mask_halfwidth+1, 2*mask_halfwidth+1) will be used.

    Returns:
        corr (np.array of shape (num_rows - 2*mask_halfwidth, num_cols - 2*mask_halfwidth,
        num_cols - 2*mask_halfwidth)): Array containing the normalized cross-correlation (NCC) between patches
                                       in the two images. The value corr[r, c_l, c_r] denotes the NCC between
                                       the patch centered at (r + mask_halfwidth, c_l + mask_halfwidth) in the
                                       left image and the patch centered at
                                       (r + mask_halfwidth, c_r + mask_halfwidth) at the right image.
    """

    m_height, m_width = gray_left.shape

    # Hint: Construct a tensor of patches, where patches_left[y, x, :] contains the patch in the left image
    # centered at (x, y) in a vectorized form. This allows you to compute the mean/variance for each patch over
    # the full image using standard numpy operations

    # The numpy function np.roll (https://docs.scipy.org/doc/numpy/reference/generated/numpy.roll.html) could be
    # useful here
    patches_left = np.zeros((m_height, m_width, (2 * mask_halfwidth + 1) ** 2))
    patches_right = np.zeros((m_height, m_width, (2 * mask_halfwidth + 1) ** 2))

    # TODO: Construct patches_left and patches_right
    #
    # ...
    #

    # TODO: normalize each patch
    #
    # ...
    #

    # TODO: Compute correlation.
    # Hint: This can be computed as a matrix multiplication. Check np.matmul and np.transpose
    #
    # ...
    #

    corr = np.zeros((m_height - 2 * mask_halfwidth, m_width - 2 * mask_halfwidth, m_width - 2 * mask_halfwidth))  # replace this
    return corr


def define_points_3d(gray_left, gray_right, mask_halfwidth, camera_parameters):
    """Compute point correspondences for two images and perform 3D reconstruction.

    Args:
        gray_left (np.array of shape (num_rows, num_cols)): left grayscale image
        gray_right (np.array of shape (num_rows, num_cols)): right grayscale image
        mask_halfwidth (int): Half-size of the square neighbourhood used for computing NCC. Thus a patch of size
                              (2*mask_halfwidth+1, 2*mask_halfwidth+1) will be used.
        camera_parameters (dict): Dict containing camera parameters
    Returns:
        points3d (np.array of shape (num_rows - 2*mask_halfwidth, num_cols - 2*mask_halfwidth, 3):
            Array containing the re-constructed 3D word coordinates for each pixel in the left image (excluding the
            boundary regions, which are ignored during NCC computation).
    """

    m_height, m_width = gray_left.shape

    new_width = m_width - 2 * mask_halfwidth
    new_height = m_height - 2 * mask_halfwidth

    # TODO: Compute normalized cross correlation and find corresponding projected x-coordinates in left and right image
    #
    # ...
    #

    # TODO: Triangulate the points to get 3D world coordinates
    #
    # ...
    #

    points3d = np.zeros((new_height, new_width, 3))  # replace this
    return points3d


def plot_correlation(gray_left, gray_right, corr, col_to_plot):
    """ Plot the normalized cross-correlation for a given column. The column for which NCC is being plotted is marked
    with a red line in the left image.

    Args:
        gray_left (np.array of shape (num_rows, num_cols)): left grayscale image
        gray_right (np.array of shape (num_rows, num_cols)): right grayscale image
        corr (np.array of shape (num_rows - 2*mask_halfwidth, num_cols - 2*mask_halfwidth,
                                 num_cols - 2*mask_halfwidth): Computed normalized
            cross-correlation (NCC) between patches in the two images.
        col_to_plot: the column in the left image for which to plot the NCC
    """

    # Pad the slice so that it's size is same as the images for easier comparison.
    pad_rows = int((gray_left.shape[0] - corr.shape[0]) / 2)
    pad_cols = int((gray_left.shape[1] - corr.shape[1]) / 2)
    corr = np.pad(corr, ((pad_rows, pad_rows), (pad_cols, pad_cols), (pad_cols, pad_cols)), 'constant',
                  constant_values=0)

    corr_slice = corr[:, col_to_plot, :]

    # Draw line in the left image to denote the column being visualized
    gray_left = np.dstack([gray_left, gray_left, gray_left])
    gray_left[:, col_to_plot, 0] = 255
    gray_left[:, col_to_plot, 1] = 0
    gray_left[:, col_to_plot, 2] = 0

    plt.ion()
    f, axes_array = plt.subplots(1, 3, figsize=(18, 16))
    axes_array[0].set_title('Left camera image', fontsize=12)
    axes_array[0].imshow(gray_left, cmap=plt.cm.gray)

    axes_array[0].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')
    axes_array[1].set_title('Right camera image', fontsize=12)
    axes_array[1].imshow(gray_right, cmap=plt.cm.gray)
    axes_array[1].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')

    axes_array[2].set_title('NCC for column marked by red line', fontsize=12)
    axes_array[2].imshow(corr_slice)
    axes_array[2].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')

    plt.show()


def plot_point_cloud(gray_left, points3d):
    """ Visualize the re-constructed point-cloud

        Args:
            gray_left (np.array of shape (num_rows, num_cols)): left grayscale image
            points3d ((np.array of shape (num_rows - 2*rad_y, num_cols - 2*rad_x, 3)): 3D World co-ordinates for each
                pixel in the left image (excluding the boundary pixels which are ignored during NCC calculation).
        """

    margin_y = gray_left.shape[0] - points3d.shape[0]
    margin_x = gray_left.shape[1] - points3d.shape[1]

    points3d = points3d[5:-5, 5:-5, :]
    colors = []
    for r in range(points3d.shape[0]):
        for c in range(points3d.shape[1]):
            col = gray_left[r + margin_y, c + margin_x]
            colors.append('rgb(' + str(col) + ',' + str(col) + ',' + str(col) + ')')
    data = [go.Scatter3d(
        x=-1 * points3d[:, :, 0].flatten(),
        y=-1 * points3d[:, :, 2].flatten(),
        z=-1 * points3d[:, :, 1].flatten(),
        mode='markers',
        marker=dict(
            size=1,
            color=colors,
            line=dict(width=0)
        )
    )]
    layout = go.Layout(
        scene=dict(camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.1, y=1, z=0.1)
        )
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig = go.Figure(
        data=data,
        layout=layout
    )
    fig.show()

# ----------------------------------------------------------------------------------------------------------------

# Execute main function
if __name__ == '__main__':
    main()

