import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
from skimage.color import rgb2gray
import cv2
import pandas as pd


def rotate_image(image, angle):

    # courtesy of stack overflow
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def file_checker(name, path):
    # Creates a list of names of all the files in a folder with a keyword (given by name)
    files = os.listdir(path)
    lis = []
    for v in files:
        if name in v:
            lis.append(v)
        else:
            continue
    return lis

def dimensions_x(arr):
    new_arr = arr > 0.8
    # Creates new image of given array with grayscale values above a certain threshold
    lis = []
    var = 0
    while var < len(new_arr):
        d = 0
        for z in new_arr[var,:]:
            if z > 0.8:
                d=d+1

            elif z == False and d > 0:
                lis.append(d)
                d=0
            else:
                continue
            # Finds the longest continued length of values above a certain threshold (in this case 0.8)
        lis.append(d)
        # creates a list of the all the lengths of continuous sections with illuminance values greater than 0.8
        var = var+1
    return max(lis)
# Finds the maximum value in the whole picture

def dimensions_y(arr):
    # Same function as the dimensions_x except this one searches for column lengths instead of rows
    new_arr = arr > 0.8
    lis = []
    var = 0
    while var < len(new_arr[0]):
        d = 0
        for j,z in enumerate(new_arr[:,var]):
            if z > 0.8:
                d = d + 1

            elif z == False and d > 0:
                lis.append(d)
                d = 0
            else:
                continue

        lis.append(d)
        var = var+1
    return max(lis)

path = input('path?')
# Give directory for where the images are located
titles = file_checker('%',path)
# Picks out the specific useful images given by the keyword
fil = {'Strain': [], "Cycle":[], "Wavelength":[], "Max X":[], "Max Y":[], "Ratio(X/Y)":[]}
# Dictionary that will be used to turn into a csv file later on
for var in titles:
    strain = var.split('-')[1]
    cycle = var.split('-')[0]
    # Extract strain and cycle from the name (needs to be standardized and varies with different files)
    image = rgb2gray(io.imread(var))
    # Turn image into grayscale
    if image.shape[0] == 1600:
        image = rotate_image(image, 90)[:,:1200]
    elif image.shape[0] == 1600:
        image = image[:,:1200]
    # Some images were vertical, needed to flip them horizontally, and cut off the measurement bar (which would have affected the wavefront values because it
    # is purely white)
    xx = dimensions_x(image)
    yy = dimensions_y(image)
    # find values of the maximum length and width
    wavelength = '633nm'
    fil['Strain'].append(strain)
    fil['Cycle'].append(cycle)
    fil['Wavelength'].append(wavelength)
    fil['Max X'].append(xx)
    fil['Max Y'].append(yy)
    fil["Ratio(X/Y)"].append((xx/yy))
    # Adds the values to the dictionary
print(fil)
df = pd.DataFrame(fil)
df.to_csv('(Luminance)Wavefront.csv')
# Convert the dictionary to a csv file

