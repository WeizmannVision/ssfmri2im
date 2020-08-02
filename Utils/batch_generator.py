from keras.utils import Sequence
import numpy as np
import os
import random
from scipy.misc import imread

from Utils.image_functions import image_prepare, rand_shift
import config_file

class batch_generator_dec(Sequence):

    """
    Generates batches for decoder model

    :param X: images
    :param Y: fMRI activations
    (X,Y correspond )
    :param batch_size: batch_size
    """

    def __init__(self,  X, Y, batch_size =32):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y = Y
        self.X = X
        self.indexes  = np.random.permutation(self.Y.shape[0])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return max(int(self.Y.shape[0] // self.batch_size), 1) #
    def __getitem__(self,batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        return self.Y[indexes], self.X[indexes]

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)



class batch_generator_enc(batch_generator_dec):
    """
    Generates batches for encoder model

    :param X: images
    :param Y: fMRI activations
    (X,Y correspond )
    :param batch_size: batch_size
    :param max_shift: max random shift applied on images
    """

    def __init__(self, X, Y, batch_size=32,max_shift = 5):
        super().__init__(X, Y, batch_size)
        self.max_shift = max_shift

    def __getitem__(self,batch_num):
        y, x = super().__getitem__(batch_num)
        x_shifted = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_shifted[i] = rand_shift(x[i],self.max_shift)
        return x_shifted,y



class batch_generator_external_images(Sequence):
    """
    Generates batches of images from a directory

    :param img_size: image should be resized to size
    :param batch_size: batch size
    :param ext_dir: directory containing images
    """
    def __init__(self, img_size = config_file.image_size, batch_size=16,ext_dir = config_file.external_images_dir):
        self.img_size = img_size
        self.batch_size = batch_size

        self.img_files = os.listdir(ext_dir)
        self.ext_dir = ext_dir

    def __getitem__(self,batch_num):
        img_file = random.sample(self.img_files , self.batch_size)
        images = np.zeros([self.batch_size, self.img_size, self.img_size, 3])
        count = 0

        for file in img_file:
            img = imread(self.ext_dir + '/' + file)
            images[count] = image_prepare(img, self.img_size)
            count += 1
        return images

    def __len__(self):
        return  50000// self.batch_size



class batch_generator_test_fmri(Sequence):
    """
    Generates test fMRI samples with random averaging
    :param Y: fMRI samples
    :param labels: Y labels
    :param batch_size: batch_size
    :param frac: specify the random fraction of test fmri to average on (3 -> 1/3)
    :param ignore_labels: labels to be omitted from batches
    """
    def __init__(self,Y,labels, batch_size=32, frac =3,ignore_labels = None):
        self.Y = Y
        self.labels = labels
        self.frac = frac
        self.num_vox = Y.shape[1]
        self.batch_size = batch_size
        self.ignore_labels = ignore_labels

    def __getitem__(self,batch_num):
        y = np.zeros([self.batch_size, self.num_vox])
        for i in range(self.batch_size):
                label = np.random.choice(self.labels, 1)
                if(self.ignore_labels is not None):
                    while(label in self.ignore_labels):
                        label = np.random.choice(self.labels, 1)

                indexes = self.get_random_indexes(label, frac=self.frac)
                y[i] = np.mean(self.Y[indexes, :], axis=0, keepdims=True)

        return y

    def get_random_indexes(self,label,frac =3):
        indexes = np.where(self.labels == label)[0]
        rand_ind = np.random.choice(frac, indexes.shape)
        while (np.sum(rand_ind) == 0 ):
            rand_ind = np.random.choice(frac, indexes.shape)
        return indexes[rand_ind == 0]



class batch_generator_encdec(Sequence):
    """
    Generates batches from encoder-deocder model
    :param X: train images
    :param Y: fMRI train samples
    (X,Y correspond )
    :param Y_test: fMRI test samples
    :param test_labels: Y_test labels
    :param batch_size: batch_size
    :param max_shift_enc: max random shift applied on images given to encoder
    :param frac_test: specify the random fraction of test fmri to average on (3 -> 1/3)
    :param ext_dir: directory containing images that are not part of the dataset
    :param ignore_test_fmri_labels: test labels to be omitted from batches
    """


    def __init__(self, X, Y, Y_test, test_labels, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = config_file.image_size
                 , frac_test = 3,ext_dir = config_file.external_images_dir,ignore_test_fmri_labels = None):
        self.num_samples = Y.shape[0]
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_dec        = batch_generator_dec(X, Y, batch_size=batch_paired)
        self.gen_enc        = batch_generator_enc(X, Y, batch_size=batch_paired,max_shift = max_shift_enc)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_unpaired,ext_dir = ext_dir)
        self.gen_test_fmri  = batch_generator_test_fmri(Y_test,test_labels, batch_size=batch_unpaired, frac =frac_test,ignore_labels = ignore_test_fmri_labels)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired

    def on_epoch_end(self):
        self.gen_dec.on_epoch_end()
        self.gen_enc.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return max(int(self.num_samples // self.batch_size), 1)

    def __getitem__(self,batch_num):
        y_in, x_out =  self.gen_dec.__getitem__(batch_num)
        x_in, y_out = self.gen_enc.__getitem__(batch_num)
        x_ext = self.gen_ext_im.__getitem__(batch_num)
        y_test_avg = self.gen_test_fmri.__getitem__(batch_num)

        x_in = np.concatenate([x_in, x_ext], axis=0)
        x_out = np.concatenate([x_out,x_ext], axis=0)

        y_in = np.concatenate([y_in, y_test_avg], axis=0)
        y_out = np.concatenate([y_out, y_test_avg], axis=0)
        mode = np.concatenate([np.ones([self.batch_paired, 1]), np.zeros([self.batch_unpaired, 1])], axis=0)
        return [y_in, x_in, mode], [x_out, y_out]




