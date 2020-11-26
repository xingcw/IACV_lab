"""
Spatial sampling
Usage:
(with aliasing):
python 1_sample.py --in ~cvcourse/pics/carpet.png --out carpet_out.png --factor 4

(with no aliasing using a low pass filter):
python 1_sample.py --in ~cvcourse/pics/carpet.png --out carpet_out.png --factor 4 --sigma 2.0
"""

from __future__ import print_function
import argparse
import sys
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

# Helper functions.

def parse_args():
    """
    Parse input arguments.
    """

    parser = argparse.ArgumentParser(description='Re-sample an image.')
    parser.add_argument('--in', dest='input_file', required=True, help='Full path of the input image file.')
    parser.add_argument('--out', dest='output_file', default='out.png', help='Filename of the output image.')
    parser.add_argument('--factor', dest='factor', type=int, default=2, help='Subsampling factor. Default: 2.')
    parser.add_argument('--sigma', dest='sigma', type=float, default=0.0, help='Strength of lowpass filter. Default: 0.0')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
	
	
def read_input_image(input_file_name):
    """
    Read image from input file into a numpy.ndarray variable of two dimensions (grayscale) with 32-bit float entries.
    """

    # Set the flag argument to 0 so that the output has only one channel.
    return cv2.imread(input_file_name, 0).astype('float')


def write_output_image_to_file(output_image, output_file_name):
    """
    Save output image to a file.
    """

    cv2.imwrite(output_file_name, output_image)
    return


def check_size(input_image, factor):
    """
    Check if the subsampling factor is too large.
    """

    condition1 = ((input_image.shape[0] // factor) == 0)
    condition2 = ((input_image.shape[1] // factor) == 0)
    
    if condition1 or condition2:
        print('Error! Subsampling rate is too large.')
        return 0
    
    else:
        print('Sub-sampling factor is permissible.')
        return 1


def subsample_image(input_image, factor):
    """
    Subsample the input image with the requested subsampling factor.
    Input parameters:
        input_image: the input image
        factor: the required sub-sampling factor
    Output: 
        the sub-sampled image
    """

    # ************************************
    # TODO
    # ************************************
    # Currently, the output image is just being set to the input image.
    # Replace this with the appropriate sub-sampling code
    # Hint: You may do this using a double for loop, but there is also a way to do this with one line of code!
    # ************************************
    x = [i for i in range(input_image.shape[0]) if i % factor == 0]
    y = [i for i in range(input_image.shape[1]) if i % factor == 0]
    output_image = input_image[np.ix_(x, y)]
    return output_image
    
    
def gaussian_filter_image(input_image, sigma):
    """
    Apply a gaussian blurring to the image
    Input parameters:
        input_image: the input image
        sigma: strength of the required gaussian blurring
    Output:
        gaussian blurred image
    """

    # ************************************
    # TODO
    # ************************************
    # Currently, the output image is just being set to the input image.
    # Replace this with the appropriate code for smooth the image
    # Hint: Look at the modules being imported at the top of the file.
    # ************************************    
    output_image = gaussian_filter(input_image, sigma=sigma)
    return output_image
	

# ----------------------------------------------------------------------------------------------------------------------

# Main function.
if __name__ == '__main__':

    # Parse command-line arguments and assign each argument to a separate variable.
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    factor = args.factor
    sigma = args.sigma

    # Read input image into a variable.
    I_input = read_input_image(input_file)

    # Print the size of the input image
    print('Size of the input image: {:d}, {:d}.'.format(I_input.shape[0], I_input.shape[1]))

    # Print the requested subsampling factor
    print('Requested subsampling factor: {:d}.'.format(factor))

    # Exit if the subsampling factor is too large
    if (check_size(I_input, factor) == 0):
        exit()

    # If requested, filter the image before subsampling
    if (sigma != 0):
        print('Applying a gaussian blur to the image.')
        I_input = gaussian_filter_image(I_input, sigma)
    else:
        print('Using the original image, without gaussian blurring.')

    # Sub-sample the image
    I_output = subsample_image(I_input, factor)

    # Print the size of the subsampled image
    print('Size of the subsampled image: {:d}, {:d}.'.format(I_output.shape[0], I_output.shape[1]))

    # Save the subsampled output to the specified file in the current working directory.
    write_output_image_to_file(I_output, output_file)

    # Plot the subsampled output image in a figure.
    plt.imshow(I_output, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Output subsampled image with a factor {:d}'.format(factor), fontsize=12)
    plt.show()
