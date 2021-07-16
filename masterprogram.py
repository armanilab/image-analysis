import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import pandas as pd
import scipy.ndimage as ndi
import os
import cv2


def rotate_image(image, angle):
    # courtesy of users on stack overflow
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
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
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


def create_folder(path, name, *args):
    # Creates a folder (or multiple) given the original directory
    real_path = f'{path}\{name}'
    for a in args:
        real_path += f'\{a}'
    os.makedirs(real_path)


def move_file(list_of_old_paths, list_of_new_paths):
    # Moves a list of files from one folder to another folder
    x = 0
    while x < len(list_of_old_paths):
        os.rename(f'{list_of_old_paths[x]}',
                  f'{list_of_new_paths[x]}')
        x = x + 1


def path_list_creator(path, list_of_file_names):
    # Creates the paths to files in a folder
    lis = []
    for j in list_of_file_names:
        lis.append(f'{path}\{j}')
    return lis


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

def transform_y(arr):
    var = 0
    xa_xis = []

    while var < len(arr):
    # Scans the whole image from top row to bottom row
        # Creates a list of all the grayscale values of a particular row
        templist = arr[var, :]
        xa_xis.append(templist.max())
        # Adds the maximum value of a particular row to a new list
        var = var + 1

    x_xis = np.array(xa_xis)
    return x_xis


def average_y(arr):
    var = 0
    xa_xis = []
    while var < len(arr):
        # Scans the whole image from top row to bottom row
        average = 0
        for j, z in enumerate(arr[var, :]):
            average = average + z
        # Adds all the grayscale values of a certain row
        averageforrow = average / len(arr[0])
        # Divides the grayscale value of the x length to find the average value
        xa_xis.append(averageforrow)
        # Adds that value to a list
        var = var + 1

    x_xis = np.array(xa_xis)
    return x_xis

def graph(gray_newsample, name, cyclenumber, wavelength, strain):
    dict1 = {"Y-Axis (in pixels)": list(range(0, len(gray_newsample))),
             "Maximum Brightness": transform_y(gray_newsample),
             "Wavelength (nanometers)": [int(wavelength)] * len(gray_newsample),
             "Cycle Number": [int(cyclenumber)] * len(gray_newsample),
             "Strain Percentage (in %)": [strain] * len(gray_newsample),
             }
    df = pd.DataFrame(dict1)
    df.to_csv(f'{name} max graph.csv')
    # Create a csv file with the values and wavelength, cycle number, strain percentage

    k = average_y(gray_newsample)
    kw = k * (1 / (max(k)))
    # All values will be percentages (in fraction) of the highest value of that image
    print(kw)
    plt.plot(kw)
    plt.margins(x=1)
    plt.xlim([0, len(gray_newsample)])
    plt.ylim(0., 1.)
    plt.xlabel('Y-Axis (In Pixels)')
    plt.ylabel('Average Brightness')
    plt.show()
    # Plots the average brightness vs y axis graph

    dict2 = {"Y-Axis (in pixels)": list(range(0, len(gray_newsample))),
             "Average Brightness": kw,
             "Wavelength (nanometers)": [int(wavelength)] * len(gray_newsample),
             "Cycle Number": [int(cyclenumber)] * len(gray_newsample),
             "Strain Percentage (in %)": [strain] * len(gray_newsample),
             }
    df1 = pd.DataFrame(dict2)
    df1.to_csv(f'{name} average graph.csv')
    # Saves the average brightness csv file

    plt.plot(transform_y(gray_newsample))
    plt.margins(x=1)
    plt.xlim([0, len(gray_newsample)])
    plt.ylim(0., 1.)
    plt.xlabel('Y-Axis (In Pixels)')
    plt.ylabel('Max Brightness')
    plt.savefig(f'{name} Max vs Y-axis')
    plt.show()
    # Saves the max brightness vs y axis graph, as well as graph it

    plt.plot(kw)
    plt.margins(x=1)
    plt.xlim([0, len(gray_newsample)])
    plt.ylim(0., 1.)
    plt.xlabel('Y-Axis (In Pixels)')
    plt.ylabel('Average Brightness')
    plt.savefig(f'{name} Average vs Y-axis')
    plt.show()
    # Saves the average brightness vs y axis graph and plots it again

path = input('path?')
# Asks for the path to the folder with all the images
cyclenumber = input('cycle number?')
# Asks for the particular cycle number of the images
wavelength = input('wavelength? in nm. please dont enter nm')
# Asks for the wavelength

subfol = f'Cycle {cyclenumber} - {wavelength}nm'
# Name of the sub folder that will be created
d=0
if int(wavelength) == 1064:
    keyword = f'% - {wavelength}nm'
    # Keyword creator  ( so the program knows which files are the useful ones)
    titles = file_checker(f'{keyword}', path)
    # Titles is a list with the names of all the files

    while d < len(titles):
        namee = os.path.splitext(titles[d])[0]
        newsample = rotate_image(io.imread(f'{titles[d]}')[..., :3], -15)[90:590, 430:500, :]
        # Trims the images, (width stays the same by height might differ)
        strain = namee.split(' ')[2]
        # Strain is located in the same place so it is very easy to find
        plt.imshow(newsample)
        plt.axis('off')
        plt.savefig(f'{titles[d]}', bbox_inches='tight', pad_inches=0)
        # Saves the trimmed image

        grey_newsample = rgb2gray(newsample)
        # Converts to grayscale image
        plt.imshow(grey_newsample, cmap='gray')
        # Saves the grayscale image
        plt.show()
        graph(grey_newsample, namee, cyclenumber, wavelength, strain)

        create_folder(path, subfol, os.path.splitext(titles[d])[0])
        list_of_names = file_checker(namee, path)
        old_path_list = path_list_creator(path, list_of_names)
        new_path_list = path_list_creator(f'{path}\{subfol}\{os.path.splitext(titles[d])[0]}', list_of_names)
        move_file(old_path_list, new_path_list)
        # Moves files to created folders
        d = d + 1

elif int(wavelength) == 633:

    keyword = '%'
    titles = file_checker(f'{keyword}', path)

    while d < len(titles):
        namee = os.path.splitext(titles[d])[0]
        newsample = io.imread(f'{titles[d]}')[:,55:115,0]
        # Trims images to 60 pixels, as well as convert to grayscale by ONLY taking the red value (in an RGB image file)
        name2 = f'{namee.split("-")[0]}-{namee.split("-")[1]}'
        strain = namee.split('-')[2]




        grey_newsample = newsample.astype(float) * (1 / 255)
        # Converts a uint8 value (0-255) to a float value (0.- 1.)
        plt.imshow(grey_newsample, cmap="gray")
        plt.axis('off')
        plt.savefig(f'{namee}.tiff', bbox_inches='tight', pad_inches=0)
        # Saves grayscale image
        graph(grey_newsample, namee, cyclenumber, wavelength, strain)

        create_folder(path, subfol, name2, os.path.splitext(titles[d])[0])
        list_of_names = file_checker(namee, path)
        old_path_list = path_list_creator(path, list_of_names)
        new_path_list = path_list_creator(f'{path}\{subfol}\{name2}\{os.path.splitext(titles[d])[0]}', list_of_names)
        move_file(old_path_list, new_path_list)
        # Moves the files to their created folders
        d = d + 1
else:
    print('unknown wavelength')