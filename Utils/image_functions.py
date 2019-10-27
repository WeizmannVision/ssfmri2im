"""
Description:
General image functions
"""
import numpy as np
from scipy.misc import imresize, imsave
from scipy.ndimage import shift
import os
import scipy

def image_prepare(img,size,interpolation = 'cubic'):
    """
    Select central crop, resize and convert gray to 3 channel image

    :param img: image
    :param size: image output size
    :param interpolation: interpolation used in resize

    :return resized and croped image

    """
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



def rand_shift(img,max_shift = 0 ):
    """
    randomly shifted image

    :param img: image
    :param max_shift: image output size

    :return randomly shifted image
    """
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    img_shifted = shift(img, [x_shift, y_shift, 0], prefilter=False, order=0, mode='nearest')
    return img_shifted



def image_collage(img_arrays, rows =10, border =5,save_file = None):
    """
    create image collage for arrays of images

    :param img_arrays: list of image arrays
    :param rows: number of rows in resulting iamge collage
    :param border: border between images
    :param save_file: location for resulting image

    :return image collage
    """

    img_len =img_arrays[0].shape[2]
    array_len  = img_arrays[0].shape[0]
    num_arrays =len(img_arrays)

    cols = int(np.ceil(array_len/rows))
    img_collage = np.ones([rows * (img_len + border) + border,num_arrays *cols * (img_len + border) , 3])

    for ind in range(array_len):
        x = (ind % cols) * num_arrays
        y = int(ind / cols)

        img_collage[border * (y + 1) + y * img_len:border * (y + 1) + (y + 1) * img_len, cols * (x + 1) + x * img_len:cols * (x + 1) +(x + num_arrays) * img_len]\
            = np.concatenate([img_arrays[i][ind] for i in range(num_arrays) ],axis=1)

    if(save_file is not None):
        imsave(save_file,img_collage)

    return img_collage



def save_images(images,images_orig = None ,folder=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(images.shape[0]):
        if(images_orig is None):
            scipy.misc.imsave(folder+'img_'+str(i)+'.jpg',images[i])
        else:
            img_concat = np.concatenate([images_orig[i],images[i]],axis=1)
            img_concat = np.squeeze(img_concat)
            scipy.misc.imsave(folder + 'img_' + str(i) + '.jpg', img_concat)