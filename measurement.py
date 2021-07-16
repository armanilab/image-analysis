import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import pandas as pd

def create_circular_mask(h, w, center=None, radius=None):
    # creates array with either false or true values

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
def count_value(im):
    total = im.sum()
    # Count the total value of all the pixels (in grayscale value)
    return total
def create_mask(ima,center,radius):

    # Uses the create_circular mask function to create an image where the locations with "false" becomes zero (black)

    im = ima.copy()
    y,x = im.shape
    mask = create_circular_mask(y, x, center = center, radius = radius)
    im[~mask] = 0
    # Anywhere where the mask == False, it becomes a zero value
    new_im = im[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
    plt.imshow(new_im, cmap='gray')
    plt.show()
    e = count_value(new_im)
    return e

center0 = [870,350]
center_1 = [865,480]
center_2 = [880,610]
center_3 = [890,760]

# Chosen centers for the different diffraction locations

radius = 60
name = '4-0%R-1.bmp'
image = io.imread(name)[...,0]
# Reads only the R values
image = image.astype(float) * (1 / 255)
# Convert from uint8 to float


a = create_mask(image,center0,radius)
b = create_mask(image,center_1,radius)
c = create_mask(image,center_2,radius)
d = create_mask(image,center_3,radius)
dict = {"Total Illuminance Value":[a,b,c,d] , "Diffraction Modes": [0,-1,-2,-3]}
df = pd.DataFrame(dict)
df.to_csv('Total Illuminance Values 633nm.csv')
# CSV file creator
