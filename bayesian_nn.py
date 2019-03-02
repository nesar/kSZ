from edward.models import Normal
import tensorflow as tf
import numpy as np
import edward as ed
import matplotlib.pyplot as plt
import seaborn as sns


def build_toy_dataset(N=50, noise_std=0.1):
  x = np.linspace(-3, 3, num=N)
  y = np.cos(x) + np.random.normal(0, noise_std, size=N)
  x = x.astype(np.float32).reshape((N, 1))
  y = y.astype(np.float32)
  X = x.reshape((N, D))
  return x, y

def neural_network(x, W_0, W_1, W_2, b_0, b_1, b_2):
    h = tf.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])

N = 784 # 50 # number of data ponts
D = 6   # 1 # number of features

W_0 = Normal(loc=tf.zeros([D, 10]), scale=tf.ones([D, 10]))
W_1 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]))
W_2 = Normal(loc=tf.zeros([10, 1]), scale=tf.ones([10, 1]))
b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10))
b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10))
b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1))





#FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_integer("N", help="Number of data points.")
#tf.flags.DEFINE_integer("D", help="Number of features.")


# x_train, y_train = build_toy_dataset(N)

#
#
#
# ############## i/o ###########################################
import params

num_train = params.num_train # 512
num_test = params.num_test # 32



datafile = ['DES', 'COSMOS', 'Galacticus'][1]


if datafile == 'DES' :
  dirIn = '../data/'
  allfiles = ['DES.train.dat', './DES5yr.nfits.dat']

  Trainfiles = np.loadtxt(dirIn + allfiles[0])
  Testfiles = np.loadtxt(dirIn + allfiles[1])

  TrainshuffleOrder = np.arange(Trainfiles.shape[0])
  TestshuffleOrder = np.arange(Testfiles.shape[0])

  Trainfiles = Trainfiles[TrainshuffleOrder]
  Testfiles = Testfiles[TestshuffleOrder]

  x_train = Trainfiles[:num_train, 2:10]  # color mag
  x_test = Testfiles[:num_test, 2:10]  # color mag
  y_train = Trainfiles[:num_train, 0]  # spec z
  y_test = Testfiles[:num_test, 0]  # spec z



if datafile == 'COSMOS' :
  dirIn = '../../Data/fromJonas/'
  allfiles = ['catalog_v0.txt', 'catalog_v3.txt'][0]


  Trainfiles = np.loadtxt(dirIn + allfiles)

  TrainshuffleOrder = np.arange(Trainfiles.shape[0])

  Trainfiles = Trainfiles[TrainshuffleOrder]

  x_train = Trainfiles[:num_train, 2:8]  # color mag
  x_test = Trainfiles[num_train + 1:, 2:8]  # color mag
  y_train = Trainfiles[:num_train, 0]  # spec z
  y_test = Trainfiles[num_train + 1:, 0]  # spec z




print("Size of features in training data: {}".format(x_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(x_test.shape))
print("Size of output in test data: {}".format(y_test.shape))
sns.regplot(x_train[:,1], y_train, fit_reg=False)
plt.show()


#---------------------------------------------------------------



print(x_train.shape)
print(y_train.shape)

x = tf.cast(x_train, dtype=tf.float32)
y = Normal(loc=neural_network(x, W_0, W_1, W_2, b_0, b_1, b_2), scale=0.1 * tf.ones(N))

x = tf.placeholder(tf.float32, [N, D])
y = Normal(loc=neural_network(x, W_0, W_1, W_2, b_0, b_1, b_2), scale=0.1 * tf.ones(N), name="y")




  # INFERENCE
with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
      loc = tf.get_variable("loc", [D, 10])
      scale = tf.nn.softplus(tf.get_variable("scale", [D, 10]))
      qW_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_1"):
      loc = tf.get_variable("loc", [10, 10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10, 10]))
      qW_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_2"):
      loc = tf.get_variable("loc", [10, 1])
      scale = tf.nn.softplus(tf.get_variable("scale", [10, 1]))
      qW_2 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_0"):
      loc = tf.get_variable("loc", [10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10]))
      qb_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_1"):
      loc = tf.get_variable("loc", [10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10]))
      qb_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_2"):
      loc = tf.get_variable("loc", [1])
      scale = tf.nn.softplus(tf.get_variable("scale", [1]))
      qb_2 = Normal(loc=loc, scale=scale)

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                       W_1: qW_1, b_1: qb_1,
                       W_2: qW_2, b_2: qb_2}, data={x: x_train, y: y_train})
# inference.run(logdir='log')

inference.run(n_iter = 1000)


# Sample functions from variational model to visualize fits.
# rs = np.random.RandomState(0)
# inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
# x = tf.expand_dims(inputs, 1)
# mus = tf.stack([neural_network(x, qW_0.sample(), qW_1.sample(), qW_2.sample(), qb_0.sample(),
#                 qb_1.sample(), qb_2.sample())
#                 for _ in range(100)])


rs = np.random.RandomState(0)
inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
x = tf.expand_dims(inputs, 1)
mus = tf.stack([neural_network(x, qW_0.sample(), qW_1.sample(), qW_2.sample(), qb_0.sample(),
                qb_1.sample(), qb_2.sample())
                for _ in range(100)])


outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 1000")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()
