from __future__ import print_function
import tensorflow as tf
from astropy.io import fits
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Lambda, MaxPooling2D, Input
from keras.layers.convolutional import Convolution2D
from keras import backend as K
import os
from keras.optimizers import RMSprop, Adam, Adadelta
#from keras.utils import multi_gpu_model


if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TF-native augmentation APIs')

K.set_image_dim_ordering('tf')

snap_array = np.array([10,11,12,14])
#snap_redshift = np.array([1.04,1.32,1.59,1.84])

batch_size = 32
num_para = 1
epochs = 100
filenum = 1000
test_split = 0.2
image_size = 256 #1024

learning_rate = 0.0001
decay_rate = 0.02
G_num = 4


save_dir = '/pylon5/as5phsp/yuyuw/kSZ/saved_models/'
model_name = 'ksz_Zsnap_higgs_Udropout_full'

# The data, split between train and test sets:


# ksz_matrix = np.zeros((filenum, image_size, image_size))
# tsz_matrix = np.zeros((filenum, image_size, image_size))

k_matrix = np.zeros((filenum*len(snap_array), image_size, image_size,2))
#tsz_matrix = np.zeros((filenum*len(snap_array), image_size, image_size))
velocity_array = np.zeros((filenum*len(snap_array)))


for ii in range(len(snap_array)):
    snap = snap_array[ii]
    f1 = open("/pylon5/as5phsp/yuyuw/kSZ/Cluster/cluster_new_0"+str(snap)+".dat")
    lines2 = f1.readlines()
    data = [line.split() for line in lines2]
    #data = data[1:]
    data = list([list(map(float, line)) for line in data])
    data = np.array(data)
    index = [i for i,j in enumerate(data[:,11]) if j>1000]
    data = data[index]
    velocity_array[ii*filenum:(ii+1)*filenum] = data[:filenum,6]
    f1.close()

    for fi in range (filenum):
        # fileIn = "mymap_box0_s14_new_"+str(fi)+".014.a.z.fits"
        fileIn = "/pylon5/as5phsp/yuyuw/kSZ/maps/Zsnap_Box0/mymap_box0_s"+str(snap)+"_is256_Zsnap_"+str(fi)+".0"+str(snap)+".a.z.fits"
        
        f0=fits.open(fileIn)
        k_matrix[ii*filenum+fi,:,:,0] = f0[1].data
        f0.close()


## CROPPING IMAGE to 128x128
# image_size = 128
# ksz_matrix = ksz_matrix[:, 480: 480 +image_size, 480: 480+ image_size]
###########################

num_train = int((1-test_split)*filenum)

np.random.seed(1234)
shuffleOrder = np.arange(filenum)
np.random.shuffle(shuffleOrder)
k_matrix = k_matrix[shuffleOrder]
#kt_matrix = np.arcsinh(kt_matrix)
# tsz_matrix = tsz_matrix[shuffleOrder]
velocity_array = velocity_array[shuffleOrder]
#allPara = np.dstack((ksz_matrix, tsz_matrix))[0]
#print (allPara.shape)

#_test_k = tf.convert_to_tensor(k_matrix[num_train:filenum], np.float64)
#x_test_t = tf.convert_to_tensor(t_matrix[num_train:filenum], np.float64)
#y_test = tf.convert_to_tensor(velocity_array[num_train:filenum], np.float64)

x_train = k_matrix[0:num_train]
y_train = velocity_array[0:num_train]

x_test = k_matrix[num_train:filenum]
y_test = velocity_array[num_train:filenum]

#print (len(k_matrix), len(x_train_k), len(x_test_k))

#input_train_k = tf.convert_to_tensor(k_matrix[0:num_train], np.float64)
#input_train_t = tf.convert_to_tensor(t_matrix[0:num_train], np.float64)
#input_train_y = tf.convert_to_tensor(velocity_array[0:num_train], np.float64)

input_shape = x_train.shape[1:]
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
output_a = Input(shape=(1,), dtype='float64')

ksz = Conv2D(32, (3, 3), padding='same',activation='linear', input_shape=(image_size, image_size,1))(input_a)
ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz)                            #32 is number of kernell, size of the kernell
ksz = MaxPooling2D(pool_size=(2, 2))(ksz)                  #256*256 to 128*128
ksz = Dropout(0.25)(ksz)

ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = MaxPooling2D(pool_size=(2, 2))(ksz)
ksz = Dropout(0.25)(ksz)

ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = MaxPooling2D(pool_size=(2, 2))(ksz)
ksz = Dropout(0.25)(ksz)

ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = MaxPooling2D(pool_size=(2, 2))(ksz)
ksz = Dropout(0.25)(ksz)
ksz = Flatten()(ksz)

#x = Flatten()(x)
ksz = Dense(2048, activation='linear')(ksz)
ksz = Dense(512, activation='linear')(ksz)
ksz = Dropout(0.25)(ksz)
ksz = Dense(128, activation='linear')(ksz)
ksz = Dense(32, activation='linear')(ksz)
ksz = Dropout(0.25)(ksz)
ksz = Dense(8, activation='linear')(ksz)
ksz = Dense(4, activation='linear')(ksz)
ksz = Dense(num_para, activation='linear')(ksz)

# initiate RMSprop optimizer
opt = keras.optimizers.adam(lr=learning_rate, decay=decay_rate)

model = Model(inputs=input_a, outputs=ksz)
# Let's train the model using RMSprop

#model = multi_gpu_model(model, gpus=G_num)

model.compile(loss='mean_squared_error', optimizer=opt)
print(model.summary())

ModelFit = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

class KerasDropoutPrediction(object):
    def __init__(self,model):
        self.f = K.function(
                [model.layers[0].input, 
                 K.learning_phase()],
                [model.layers[-1].output])
    def predict(self,x, n_iter=10):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x , 1]))
        result = np.array(result).reshape(n_iter,len(x)).T
        return result


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


############################################ TESTING #########################################

#import h5py
#f = h5py.File(model_path, 'r+')
#del f['optimizer_weights']
#f.close()
#
#from keras.models import load_model
#model = load_model(model_path)

#kdp = KerasDropoutPrediction(model)
#y_pred_do = kdp.predict(x_test,n_iter=10)
#y_pred_do_mean = y_pred_do.mean(axis=1)
#y_pred_do_std = y_pred_do.std(axis=1)
#
#print (y_pred_do)
#print (y_pred_do_mean)
#print (y_test)
#print (y_pred_do_std)


# Score trained model.
#scores = model.evaluate([x_test_k,x_test_t], y_test, verbose=1)
#print('Test loss:', scores)
## print('Test accuracy:', scores[1])
#
#y_pred = model.predict([x_test_k,x_test_t])
##print (y_pred)
#y_pred = y_pred.T
#
#diff = np.abs(y_pred[0] - y_test)
#
#print('Difference min, max, mean, std, median')
#print ("least squares root", np.sqrt(np.sum(diff**2)))
#print(np.min(diff), np.max(diff), np.mean(diff), np.std(diff), np.median(diff))
#print (np.average(np.abs(diff/y_test)))
#
#epoch_array = range(1, epochs + 1)
#train_loss = ModelFit.history['loss']
#val_loss = ModelFit.history['val_loss']
##train_acc = ModelFit.history['acc']
##val_acc = ModelFit.history['val_acc']
#
#loss_inf = np.vstack((epoch_array, train_loss, val_loss))
#
#result_inf = np.vstack((y_pred[0], y_test))
#
#np.save(save_dir+model_name+"-loss_e"+str(epochs)+".npy", loss_inf)
#np.save(save_dir+model_name+"-pred_e"+str(epochs)+".npy", result_inf)
#np.save(save_dir+model_name+"-order_e"+str(epochs)+".npy", shuffleOrder)


# -------------------------------------------------


#plotLossAcc = False
#if plotLossAcc:
#    import matplotlib.pylab as plt
#
#    train_loss = ModelFit.history['loss']
#    val_loss = ModelFit.history['val_loss']
#    train_acc = ModelFit.history['acc']
#    val_acc = ModelFit.history['val_acc']
#    epoch_array = range(1, epochs + 1)
#
#    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
#    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.02)
#    ax[0].plot(epoch_array, train_loss)
#    ax[0].plot(epoch_array, val_loss)
#    ax[0].set_ylabel('loss')
#    # ax[0].set_ylim([0,1])
#    # ax[0].set_title('Loss')
#    ax[0].legend(['train_loss', 'val_loss'])
#
#    ax[1].plot(epoch_array, train_acc)
#    ax[1].plot(epoch_array, val_acc)
#    ax[1].set_ylabel('acc')
#    # ax[0].set_ylim([0,1])
#    # ax[0].set_title('Loss')
#    ax[1].legend(['train_acc', 'val_acc'])
#
#    plt.show()
#
#plotLoss = True
#if plotLoss:
#    import matplotlib.pylab as plt
#
#    train_loss = ModelFit.history['loss']
#    val_loss = ModelFit.history['val_loss']
#    epoch_array = range(1, epochs + 1)
#
#    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7, 5))
#    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.02)
#    ax.plot(epoch_array, train_loss)
#    ax.plot(epoch_array, val_loss)
#    ax.set_ylabel('loss')
#    # ax[0].set_ylim([0,1])
#    # ax[0].set_title('Loss')
#    ax.legend(['train_loss', 'val_loss'])
#
#    plt.show()
#
#test_array = np.arange(np.min(y_test), np.max(y_test), 1)
#y_pred=y_pred.T
#
## ----------- Plot prediction vs truth -----------
#ScatterPredReal = True
#if ScatterPredReal:
#    diff = np.abs(y_pred[0] - y_test)
#
#    print('Difference min, max, mean, std, median')
#    print ("least squares root", np.sqrt(np.sum(diff**2)))
#    print(np.min(diff), np.max(diff), np.mean(diff), np.std(diff), np.median(diff))
#    print (np.average(np.abs(diff/y_test)))
#
#    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#    fig.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
#    ax.scatter(y_pred[0], y_test)
#    ax.plot(test_array, test_array, c='r')
#    ax.set_ylabel('y_test')
#    ax.set_xlabel('y_pred')
#
#    plt.show()
#    
#fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#fig.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
#ax.scatter(y_test, y_pred[0]-y_test)
#ax.axhline(y=0, c='r')
#ax.set_xlabel('y_test')
#ax.set_ylabel('y_pred - y_test')
#plt.show()

# ScatterPred = True
# if ScatterPred:
#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     fig.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
#     ax.scatter(y_test, y_pred / y_test)
#     ax.set_ylabel('pred/test')
#     ax.set_xlabel('y_test')
#
#
#     plt.show()

# JointDistribution = True
# if JointDistribution:
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     fig.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
#     ax.scatter(y_test[:, 0], y_test[:, 1], label='y_test')
#     ax.set_ylabel('m200')
#     ax.set_xlabel('r200')
#     # ax.set_title()
#
#     ax.scatter(y_pred[:, 0], y_pred[:, 1], label='y_pred')
#     ax.set_ylabel('m200')
#     ax.set_xlabel('r200')
#     ax.set_title('Joint distribution')
#     plt.legend()
#
#     plt.show()


