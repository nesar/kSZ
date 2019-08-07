from keras.models import load_model
from keras import backend as K
import os
import numpy as np
from astropy.io import fits

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

snap_array = np.array([10,11,12,14])
filenum = 10000
test_split = 0.2
image_size = 256 #1024
save_dir = '/pylon5/as5phsp/yuyuw/kSZ/saved_models'
#model_name = 'ksz_Zsnap_higgs_Udropout_'+str(snap)
model_name = 'ktsz_Zsnap_Xsede_new_snapnum4epoch-100-new'

model_path = os.path.join(save_dir, model_name)
model = load_model(model_path)

k_matrix = np.zeros((filenum*len(snap_array), image_size, image_size, 1))
t_matrix = np.zeros((filenum*len(snap_array), image_size, image_size, 1))
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
        fileIn = "/pylon5/as5phsp/yuyuw/kSZ/maps/Zsnap_Box0/mymap_box0_s"+str(snap)+"_is256_Zsnap_"+str(fi)+".0"+str(snap)+".a.z.fits"
        fileIn_tsz = "/pylon5/as5phsp/yuyuw/kSZ/maps/Zsnap_Box0/mymap_box0_s"+str(snap)+"_is256_Zsnap_tSZ_"+str(fi)+".0"+str(snap)+".a.z.fits"
        
        f0=fits.open(fileIn)
        k_matrix[ii*filenum+fi,:,:,0] = f0[1].data*1000000
        f0.close()
    
        f0=fits.open(fileIn_tsz)
        t_matrix[ii*filenum+fi,:,:,0] = f0[1].data*1000000
        f0.close()


## CROPPING IMAGE to 128x128
# image_size = 128
# ksz_matrix = ksz_matrix[:, 480: 480 +image_size, 480: 480+ image_size]
###########################

num_train = int((1-test_split)*filenum)

np.random.seed(1234)
shuffleOrder = np.arange(filenum*len(snap_array))
np.random.shuffle(shuffleOrder)
k_matrix = k_matrix[shuffleOrder]
t_matrix = t_matrix[shuffleOrder]
#kt_matrix = np.arcsinh(kt_matrix)
# tsz_matrix = tsz_matrix[shuffleOrder]
velocity_array = velocity_array[shuffleOrder]
#allPara = np.dstack((ksz_matrix, tsz_matrix))[0]
#print (allPara.shape)

#_test_k = tf.convert_to_tensor(k_matrix[num_train:filenum], np.float64)
#x_test_t = tf.convert_to_tensor(t_matrix[num_train:filenum], np.float64)
#y_test = tf.convert_to_tensor(velocity_array[num_train:filenum], np.float64)

x_test_k = k_matrix[num_train:]
x_test_t = t_matrix[num_train:]
y_test = velocity_array[num_train:]

del k_matrix, t_matrix, velocity_array

kdp = KerasDropoutPrediction(model)
y_pred = np.zeros(len(y_test))
y_err = np.zeros(len(y_test))

test_batch = 50
for ii in range (len(y_test)/test_batch):
    print (ii)
    y_pred_do = kdp.predict([x_test_k[test_batch*ii:(ii+1)*test_batch], x_test_t[test_batch*ii:(ii+1)*test_batch]],n_iter=100)
    y_pred_do_mean = y_pred_do.mean(axis=1)
    y_pred[test_batch*ii:(ii+1)*test_batch]=y_pred_do_mean
    y_pred_do_std = y_pred_do.std(axis=1)
    y_err[test_batch*ii:(ii+1)*test_batch]=y_pred_do_std

#print (y_pred_do)
#print (y_pred_do_mean)
#print (y_test[:200])
#print (y_pred_do_std)

result_inf = np.vstack((y_pred, y_test, y_err))
np.save(save_dir+model_name+"-Upred.npy", result_inf)