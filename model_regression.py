import tensorflow as tf
import numpy as np
import h5py
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import keras 
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop,Adam,Nadam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from IPython.display import clear_output
from keras.layers import DepthwiseConv2D, Reshape, Activation

print(tf.__version__)


img_width, img_height, img_num_channels = 150, 150, 1
validation_split = 0.2
verbosity = 1


h5f_train = h5py.File('X_train.h5','r')
h5f_real = h5py.File('Y_train.h5','r')
h5f_test = h5py.File('X_test.h5','r')
train_data = h5f_train['train_data'][:]
real_data = h5f_real['real_data'][:]
test_data = h5f_test['test_data'][:]
h5f_train.close()
h5f_test.close()

m = real_data.shape[0]
m_test = test_data.shape[0]
#train_data =  train_data.reshape(train_data.shape[0], -1).T/255.0

# Step 1: Shuffle (X, Y)
#permutation = list(np.random.permutation(m))
#train_data = train_data[:, permutation]
#real_data = real_data[:, permutation].reshape((1, m))
   
#train_dataset = tf.data.Dataset.from_tensor_slices(
#      (train_data, real_data.T)).shuffle(13404).repeat().batch(batch_size)

train_data = train_data.reshape((m, img_width, img_height, img_num_channels))/255.0
test_data  = test_data.reshape((m_test, img_width, img_height, img_num_channels))/255.0



model = Sequential()

#### Input Layer ####
model.add(Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal',padding='same',
                 activation='relu', input_shape=(150,150, 1)))

#### Convolutional Layers ####
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))  # Pooling
model.add(Dropout(0.2)) # Dropout

#model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
#model.add(Conv2D(64, (3,3), activation='relu',kernel_regularizer=regularizers.l2(l=0.01)))
#model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu',kernel_regularizer=regularizers.l2(l=0.02)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, (5,5), padding='same', activation='relu'))
model.add(Conv2D(512, (5,5), activation='relu'))
model.add(MaxPooling2D((4,4)))
model.add(Dropout(0.2))

#### Fully-Connected Layer ####
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1))
batch_size = 64
epochs= 35

#opt = keras.optimizers.SGD(lr=0.0005, decay=1e-6, momentum = 0.9, nesterov=True)
optimizer = keras.optimizers.RMSprop(lr = 0.00008, decay = 1e-6)





model.compile(optimizer=optimizer, loss='mean_squared_error')






learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2,verbose=1,factor=0.6, min_lr=0.0000001)





filepath="weights.best.hdf5"




checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')



callbacks_list = [checkpoint,learning_rate_reduction]

history = model.fit(train_data,real_data,epochs=35,shuffle = True, callbacks = callbacks_list)



history


yhat = model.predict(test_data)
with open('predict.txt','w') as f:
    lis = yhat.tolist()
    lis = sum(lis, [])
    for item in lis:
        f.write("%s\n" % item)


my_file = open("predict.txt", "r")
content = my_file.read()
content_p = content.split("\n")
my_file.close()


rows = []

fields = ['Id', 'Energy']

for filename in listdir('/Users/kuntalpal/Downloads/mlhep2021-classification/test'):
	rows.append(filename[:-4])


rows_csv = []

for i in range(16560):
	rows_csv.append([rows[i],content_p[i]])

filename = "regression_kuntal_final.csv"

with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows_csv)