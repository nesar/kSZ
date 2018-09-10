from __future__ import print_function
import tensorflow as tf
from astropy.io import fits
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras import backend as K
import os
from keras.optimizers import RMSprop, Adam, Adadelta


if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TF-native augmentation APIs')

K.set_image_dim_ordering('tf')


batch_size = 2
num_para = 1
epochs = 50
filenum = 1000
test_split = 0.2
num_para = 1
image_size = 256 #1024

learning_rate = 0.0001
decay_rate = 0.1


save_dir = '../saved_models/'
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:


ksz_matrix = np.zeros((filenum, image_size, image_size))
tsz_matrix = np.zeros((filenum, image_size, image_size))
velocity_array = np.zeros((filenum))

f1 = open("wmap.014.cluster.sim.dat")
lines2 = f1.readlines()
data = [line.split() for line in lines2]
data = data[1:]
data = np.array([map(float, line) for line in data])
velocity_array = data[:filenum,6]
f1.close()

for fi in range (filenum):
    # fileIn = "mymap_box0_s14_new_"+str(fi)+".014.a.z.fits"
    fileIn = "Box0/mymap_box0_s14_is256_new_"+str(fi)+".014.a.z.fits"


    f0=fits.open(fileIn)
    ksz_matrix[fi] = f0[1].data
    f0.close()

    #f0=fits.open("Box0/mymap_box0_s14_tSZ_"+str(fi)+".014.a.z.fits")
    #tsz_matrix[fi] = f0[1].data
    #f0.close()


## CROPPING IMAGE to 128x128
# image_size = 128
# ksz_matrix = ksz_matrix[:, 480: 480 +image_size, 480: 480+ image_size]
###########################

num_train = int((1-test_split)*filenum)

np.random.seed(1234)
shuffleOrder = np.arange(filenum)
np.random.shuffle(shuffleOrder)
ksz_matrix = ksz_matrix[shuffleOrder]
ksz_matrix = np.arcsinh(1000000*ksz_matrix)
#tsz_matrix = tsz_matrix[shuffleOrder]
velocity_array = velocity_array[shuffleOrder]
#allPara = np.dstack((ksz_matrix, tsz_matrix))[0]
#print (allPara.shape)

x_train = ksz_matrix[0:num_train]
y_train = velocity_array[0:num_train]

x_test = ksz_matrix[num_train:filenum]
y_test = velocity_array[num_train:filenum]



# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')


#x_train -= np.min(ksz_matrix)
#x_test -= np.min(ksz_matrix)
#
#normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
#
#
#x_train /= normFactor
#x_test /= normFactor
#
#x_train = np.expand_dims(x_train, axis=1)
#y_train = np.expand_dims(y_train, axis=1)
#
#
#
#y_train -= np.min(velocity_array)
#y_test -= np.min(velocity_array)
#
#y_normFactor = np.max( [np.max(y_train), np.max(y_test ) ])
#
#y_train /= y_normFactor
#y_test /= y_normFactor

# x_train = np.rollaxis(x_train, 2, 0)
# x_train = np.rollaxis(x_train, 1, 0)
# x_train = np.rollaxis(x_train, 3, 2)


x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1)
x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()

model.add(Conv2D(32, (3, 3),
                        border_mode='same',
                        input_shape=(image_size, image_size, 1) ))

#
model.add(Activation('relu'))


model.add(Conv2D(32, (3, 3)))                               #32 is number of kernell, size of the kernell
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))                   #256*256 to 128*128
# model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# #
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# #

model.add(Flatten())
model.add(Dense(10000))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_para))
model.add(Activation('linear'))


input_shape= x_train[0].shape


# initiate RMSprop optimizer
opt = keras.optimizers.adam(lr=learning_rate, decay=decay_rate)

# Let's train the model using RMSprop
model.compile(loss='mean_squared_error', optimizer=opt)


print(model.summary())

ModelFit = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


############################################ TESTING #########################################

from keras.models import load_model
model = load_model(model_path)


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores)
# print('Test accuracy:', scores[1])

y_pred = model.predict(x_test)

# -------------------------------------------------


plotLossAcc = False
if plotLossAcc:
    import matplotlib.pylab as plt

    train_loss = ModelFit.history['loss']
    val_loss = ModelFit.history['val_loss']
    train_acc = ModelFit.history['acc']
    val_acc = ModelFit.history['val_acc']
    epoch_array = range(1, epochs + 1)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.02)
    ax[0].plot(epoch_array, train_loss)
    ax[0].plot(epoch_array, val_loss)
    ax[0].set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[0].legend(['train_loss', 'val_loss'])

    ax[1].plot(epoch_array, train_acc)
    ax[1].plot(epoch_array, val_acc)
    ax[1].set_ylabel('acc')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[1].legend(['train_acc', 'val_acc'])

    plt.show()

plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    train_loss = ModelFit.history['loss']
    val_loss = ModelFit.history['val_loss']
    epoch_array = range(1, epochs + 1)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7, 5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.02)
    ax.plot(epoch_array, train_loss)
    ax.plot(epoch_array, val_loss)
    ax.set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train_loss', 'val_loss'])

    plt.show()

# ----------- Plot prediction vs truth ----------- 
ScatterPredReal = True
if ScatterPredReal:
    diff = np.abs(y_pred - y_test)

    print('Difference min, max, mean, std, median')
    print(np.min(diff), np.max(diff), np.mean(diff), np.std(diff), np.median(diff))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    ax.scatter(y_pred, y_test)
    ax.set_ylabel('y_test')
    ax.set_xlabel('y_pred')

    plt.show()

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



