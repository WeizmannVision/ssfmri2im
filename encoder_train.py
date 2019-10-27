import copy
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.backend.tensorflow_backend import set_session
from keras.losses import mean_squared_error, cosine_proximity
from keras.optimizers import SGD

from Utils.batch_generator import *

from Utils.misc import download_network_weights
import sys


from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler

from Models.models import *
import config_file

if (os.path.exists(config_file.encoder_weights) and not config_file.retrain_encoder):
    print('pretrained encoder weights file exist')
    sys.exit()
else:
    print('training encoder')

os.environ["CUDA_VISIBLE_DEVICES"] = config_file.GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
set_session(tf.Session(config=gpu_config))


#################################################### data load ##########################################################

handler = data_handler(matlab_file = config_file.kamitani_data_mat)
Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC')
labels_train, labels = handler.get_labels()
NUM_VOXELS = Y.shape[1]

file= np.load(config_file.images_npz)
X = file['train_images']
X_test = file['test_images']

X= X[labels_train]
X_test_sorted = X_test
X_test = X_test[labels]


#################################################### losses & schedule ##########################################################

initial_lrate = 0.1


def step_decay(epoch):

   lrate = 0.1
   if(epoch>20):
       lrate = 0.01
   if (epoch > 35):
       lrate = 0.001
   if (epoch > 45):
       lrate = 0.0001
   if (epoch > 50):
       lrate = 0.00001

   return lrate

def combined_loss(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) +  0.1*cosine_proximity(y_true, y_pred)

#################################################### model ##########################################################
if (~os.path.exists(config_file.caffenet_models_weights)):
    download_network_weights()
else:
    print("pre-trained matconvnet model is ready")

enc_param = encoder_param(NUM_VOXELS)
enc_param.drop_out = 0.5

vision_model = encoder(enc_param)
vision_model.compile(loss=combined_loss, optimizer= SGD(lr=initial_lrate,decay = 0.0 , momentum = 0.9,nesterov=True),metrics=['mse','cosine_proximity','mae'])
print(vision_model.summary())


#################################################### callbacks ##########################################################
callbacks = []

if(config_file.encoder_tenosrboard_logs is not None):
    log_path = config_file.encoder_tenosrboard_logs
    tb_callback = TensorBoard(log_path)
    tb_callback.set_model(vision_model)
    callbacks.append(tb_callback)

reduce_lr = LearningRateScheduler(step_decay)
callbacks.append(reduce_lr)

#################################################### train & save ##########################################################
train_generator = batch_generator_enc(X, Y, batch_size=64,max_shift = 5)
test_generator = batch_generator_enc(X_test_sorted, Y_test_avg, batch_size=50,max_shift = 0)

vision_model.fit_generator(train_generator, epochs=80,validation_data=test_generator ,verbose=2,use_multiprocessing=False,callbacks=callbacks) #, steps_per_epoch=1200//64 , validation_steps=1
if(config_file.encoder_weights is not None):
    vision_model.save_weights(config_file.encoder_weights)

