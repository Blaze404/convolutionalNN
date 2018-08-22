# imports:
# rgb to gray from scikit-image ( skimage )
# numpy
# Image from Python Image Library to read image

from skimage.color import rgb2gray
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):

    # a function which takes an address of an image as parameter
    # convert it to an rgb based numpy array
    # and returns that array

    # parameters:
    # 1. image_path : path of the image to be read

    img = Image.open(image_path)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def convert_rgb_to_gray(img):

    # a function to convert a numpy array containing rgb values of an image
    # to gray using rgb2gray method of skimage.color

    # parameters:
    # 1. img: numpy array of rgb based image

    rgb_image = rgb2gray(img)
    return rgb_image

def get_gray_image(image_path):
    img = load_image(image_path)
    rgb_image = convert_rgb_to_gray(img)
    return rgb_image
#
# print()
i1 = load_image('images/human.jpg')

i2 = convert_rgb_to_gray(i1)
print(i1.shape)
plt.imshow(i2)
plt.show()
print(i2.shape)
# convert_rgb_to_gray(load_image('images/human.jpg'))