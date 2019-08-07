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
from keras.utils import multi_gpu_model
from keras import activations, initializers
from keras.layers import Layer
import tqdm


if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TF-native augmentation APIs')

K.set_image_dim_ordering('tf')

snap_array = np.array([10])


batch_size = 32
num_para = 1
epochs = 15
filenum = 1000
test_split = 0.2
num_para = 1
image_size = 256 #1024

learning_rate = 0.001
decay_rate = 0.002
noise = 0.0

num_batches = filenum*len(snap_array)/batch_size
kl_loss_weight = 1.0 / num_batches

save_dir = '/data0/yuyu/saved_models/'
model_name = 'ksz_only_uncertainty'+str(len(snap_array))+"epoch-"+str(epochs)

def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
    params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma

def log_mixture_prior_prob(w):
    comp_1_dist = tf.distributions.Normal(0.0, prior_params[0])
    comp_2_dist = tf.distributions.Normal(0.0, prior_params[1])
    comp_1_weight = prior_params[2]    
    return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))    

# Mixture prior parameters shared across DenseVariational layer instances
prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=0.1, pi=0.2)

class DenseVariational(Layer):
    def __init__(self, output_dim, kl_loss_weight, activation=None, **kwargs):
        self.output_dim = output_dim
        self.kl_loss_weight = kl_loss_weight
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):  
        self._trainable_weights.append(prior_params) 

        self.kernel_mu = self.add_weight(name='kernel_mu', 
                                         shape=(input_shape[1], self.output_dim),
                                         initializer=initializers.normal(stddev=prior_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu', 
                                       shape=(self.output_dim,),
                                       initializer=initializers.normal(stddev=prior_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                          shape=(input_shape[1], self.output_dim),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho', 
                                        shape=(self.output_dim,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
                
        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) + 
                      self.kl_loss(bias, self.bias_mu, bias_sigma))
        
        return self.activation(K.dot(x, kernel) + bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def kl_loss(self, w, mu, sigma):
        variational_dist = tf.distributions.Normal(mu, sigma)
        return kl_loss_weight * K.sum(variational_dist.log_prob(w) - log_mixture_prior_prob(w))
        
def neg_log_likelihood(y_true, y_pred, sigma=noise):
    dist = tf.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_true))

# The data, split between train and test sets:


ksz_matrix = np.zeros((filenum*len(snap_array), image_size, image_size))
tsz_matrix = np.zeros((filenum*len(snap_array), image_size, image_size))
velocity_array = np.zeros((filenum*len(snap_array)))


for ii in range(len(snap_array)):
    snap = snap_array[ii]
    f1 = open("/data0/yuyu/Cluster/cluster_new_0"+str(snap)+".dat")
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
        fileIn = "/data0/yuyu/maps/Zsnap_Box0/mymap_box0_s"+str(snap)+"_is256_Zsnap_"+str(fi)+".0"+str(snap)+".a.z.fits"
    
    
        f0=fits.open(fileIn)
        ksz_matrix[ii*filenum+fi] = f0[1].data
        f0.close()

    #f0=fits.open("Box0/mymap_box0_s14_tSZ_"+str(fi)+".014.a.z.fits")
    #tsz_matrix[fi] = f0[1].data
    #f0.close()


## CROPPING IMAGE to 128x128
# image_size = 128
# ksz_matrix = ksz_matrix[:, 480: 480 +image_size, 480: 480+ image_size]
###########################

num_train = int((1-test_split)*filenum*len(snap_array))

np.random.seed(1234)
shuffleOrder = np.arange(filenum*len(snap_array))
np.random.shuffle(shuffleOrder)
ksz_matrix = ksz_matrix[shuffleOrder]
#ksz_matrix = ksz_matrix
#ksz_matrix = np.arcsinh(1000000*ksz_matrix)
#tsz_matrix = tsz_matrix[shuffleOrder]
velocity_array = velocity_array[shuffleOrder]
#allPara = np.dstack((ksz_matrix, tsz_matrix))[0]
#print (allPara.shape)

x_train = ksz_matrix[0:num_train]
y_train = velocity_array[0:num_train]

x_test = ksz_matrix[num_train:filenum*len(snap_array)]
y_test = velocity_array[num_train:filenum*len(snap_array)]

normFactor = np.max(np.abs(ksz_matrix))

x_train /= normFactor
x_test /= normFactor

y_normFactor = np.max(np.abs(velocity_array))

y_train /= y_normFactor
y_test /= y_normFactor

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

input_shape = x_train.shape[1:]
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
output_a = Input(shape=(1,), dtype='float64')


ksz = Conv2D(32, (3, 3), padding='same',activation='linear', input_shape=(image_size, image_size,1))(input_a)
ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz)                            #32 is number of kernell, size of the kernell
ksz = MaxPooling2D(pool_size=(2, 2))(ksz)                  #256*256 to 128*128

ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = MaxPooling2D(pool_size=(2, 2))(ksz)

ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = MaxPooling2D(pool_size=(2, 2))(ksz)

ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = Conv2D(32, (3, 3), padding = 'same',activation='linear')(ksz) 
ksz = MaxPooling2D(pool_size=(2, 2))(ksz)
ksz = Flatten()(ksz)
# #

ksz = DenseVariational(1024, kl_loss_weight=kl_loss_weight, activation='linear')(ksz)
ksz = DenseVariational(512, kl_loss_weight=kl_loss_weight, activation='linear')(ksz)
ksz = DenseVariational(128, kl_loss_weight=kl_loss_weight, activation='linear')(ksz)
ksz = DenseVariational(64, kl_loss_weight=kl_loss_weight, activation='linear')(ksz)
ksz = DenseVariational(32, kl_loss_weight=kl_loss_weight, activation='linear')(ksz)
ksz = DenseVariational(32, kl_loss_weight=kl_loss_weight, activation='linear')(ksz)
ksz = DenseVariational(8, kl_loss_weight=kl_loss_weight, activation='linear')(ksz)
ksz = DenseVariational(4, kl_loss_weight=kl_loss_weight, activation='linear')(ksz)
ksz = DenseVariational(1, kl_loss_weight=kl_loss_weight, activation='linear')(ksz)


#input_shape= x_train[0].shape

# initiate RMSprop optimizer
opt = keras.optimizers.adam(lr=learning_rate, decay=decay_rate)

model = Model(inputs=input_a, outputs=ksz)

# Let's train the model using RMSprop
model.compile(loss='mean_squared_error', optimizer=opt)
#model.compile(loss=neg_log_likelihood, optimizer=opt, metrics=['mse'])

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

#from keras.models import load_model
#model = load_model(model_path)


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores)
# print('Test accuracy:', scores[1])

y_pred_list = []
for i in tqdm.tqdm(range(50)):
    y_pred = model.predict(x_test)
    y_pred_list.append(y_pred)

y_preds = np.concatenate(y_pred_list, axis=1)

y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)

print (y_test, y_preds, y_mean, y_sigma)


#y_pred = model.predict(x_test)
#y_pred = y_pred.T

#epoch_array = range(1, epochs + 1)
#train_loss = ModelFit.history['loss']
#val_loss = ModelFit.history['val_loss']
##train_acc = ModelFit.history['acc']
##val_acc = ModelFit.history['val_acc']
#
#loss_inf = np.vstack((epoch_array, train_loss, val_loss))
#
#result_inf = np.vstack((y_preds, y_test))
#
#np.save(save_dir+model_name+"-loss.npy", loss_inf)
#np.save(save_dir+model_name+"-pred.npy", result_inf)

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

#test_array = np.arange(np.min(y_test), np.max(y_test), 1)
#y_pred = y_pred.T
#
## ----------- Plot prediction vs truth ----------- 
#ScatterPredReal = True
#if ScatterPredReal:
#    diff = np.abs(y_pred[0] - y_test)
#
#    print('Difference min, max, mean, std, median')
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



