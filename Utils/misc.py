import numpy as np
import scipy.stats as stat
import os
from six.moves import urllib
from Utils.image_functions import image_collage
from scipy.misc import imsave
import keras
import config_file

def calc_snr(y, y_avg, labels):
    sig = np.var(y_avg, axis=0)
    noise = 0
    for l in labels:
        noise += np.var(y[labels == l], axis=0)
    noise /= len(labels)
    return sig/noise



def corr_percintiles(y,y_pred, per = [50,75,90]):
    num_voxels = y.shape[1]
    corr = np.zeros([num_voxels])

    for i in range(num_voxels):
        corr[i] = stat.pearsonr(y[:, i], y_pred[:, i])[0]
    corr = np.nan_to_num(corr)

    corr_per = []
    for p in per:
        corr_per.append(np.percentile(corr,p))
    return corr_per

class log_image_collage_callback(keras.callbacks.Callback):
    def __init__(self, Y, X, model, dir = '',freq = 10):
        self.Y = Y
        self.X = X
        self.pred_model = model
        self.freq = freq
        self.dir = dir

    def on_epoch_end(self, epoch, logs={}):
        if(epoch%self.freq==0):
            X_pred = self.pred_model.predict(self.Y)
            collage = image_collage([self.X,X_pred])
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)


            imsave(self.dir+'ep_'+str(epoch)+'.jpg',collage)



def download_network_weights(download_link = config_file.DOWNLOAD_LINK, file_name = config_file.caffenet_models_weights, expected_bytes = config_file.EXPECTED_BYTES):
    """ Download the pretrained model if it's not already downloaded """

    print("Downloading the pre-trained model.")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)
    file_stat = os.stat(file_name)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded pre-trained model', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')