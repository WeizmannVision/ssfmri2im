import numpy as np

import pandas as pd
from scipy.misc import imread
import matplotlib.pyplot as plt
from scipy.misc import imresize
import config_file
import os


# Create Image dataset from Imagenet folders
def image_generate(imgnet_dir = config_file.imagenet_wind_dir,test_csv='./imageID_test.csv',train_csv='./imageID_training.csv',size = config_file.image_size,out_file= config_file.images_npz,interpolation = 'cubic'):
    test_im = pd.read_csv(test_csv,header=None)
    train_im = pd.read_csv(train_csv,header=None)

    test_images = np.zeros([50, size, size, 3])
    train_images = np.zeros([1200, size, size, 3])

    count = 0

    for file in list(test_im[1]):
        folder = file.split('_')[0]
        img = imread(imgnet_dir + folder + '/' + file)
        test_images[count] = image_prepare(img, size,interpolation)
        count += 1

    count = 0

    for file in list(train_im[1]):
        folder = file.split('_')[0]
        img = imread(imgnet_dir + folder + '/' + file)
        train_images[count] = image_prepare(img, size,interpolation)
        count += 1
    np.savez(out_file, train_images=train_images, test_images=test_images)


#ceneter crop and resize
def image_prepare(img,size,interpolation):

    out_img = np.zeros([size,size,3])
    s = img.shape
    r = s[0]
    c = s[1]

    trimSize = np.min([r, c])
    lr = int((c - trimSize) / 2)
    ud = int((r - trimSize) / 2)
    img = img[ud:min([(trimSize + 1), r - ud]) + ud, lr:min([(trimSize + 1), c - lr]) + lr]

    img = imresize(img, size=[size, size], interp=interpolation)
    if (np.ndim(img) == 3):
        out_img = img
    else:
        out_img[ :, :, 0] = img
        out_img[ :, :, 1] = img
        out_img[ :, :, 2] = img

    return out_img/255.0

if __name__ == "__main__":
    if(os.path.exists(config_file.images_npz)):
        print('images npz file exists')
    else:
        print('creating npz file exists')
        image_generate()
