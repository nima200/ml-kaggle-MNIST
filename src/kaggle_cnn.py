
# coding: utf-8

# In[2]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras import backend as K
import pre_processing as pp
from importlib import reload
K.set_image_dim_ordering('th')


# In[3]:


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# In[ ]:


training = pp.get_training()


# In[24]:


test = pp.get_test()


# In[25]:


X_train = np.array([x for x, y in training])
y_train = np.array([y for x, y in training])
X_test = test


# In[27]:


pixel_count = X_train.shape[1] * X_train.shape[2]


# In[28]:


# Normalize values (0-1)
X_train = np.divide(X_train, 255)
X_test = np.divide(X_test, 255)
y_train = np_utils.to_categorical(y_train)


# In[29]:


class_count = y_train.shape[1]


# In[30]:


# Seed for reproducibility
seed = 42
np.random.seed(seed)


# In[12]:


# Flatten
# X_train = X_train.reshape(X_train.shape[0], pixel_count).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], pixel_count).astype('float32')


# In[13]:


# BASE NEURAL NETWORK MODEL
# def base_nn():
#     nn_model = Sequential()
#     # Relu hidden layer, same number of neurons as input layer
#     nn_model.add(Dense(pixel_count, input_dim=pixel_count, kernel_initializer='normal', activation='relu'))
#     # Softmax output layer
#     nn_model.add(Dense(class_count, kernel_initializer='normal', activation='softmax'))
#     # Compile
#     nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return nn_model


# In[14]:


# # Create a base neural network model
# model = base_nn()
# model.fit(X_train, y_train_vec, validation_data=(X_test, y_test_vec), epochs=10, batch_size=50, verbose=2)
# scores = model.evaluate(X_test, y_test_vec, verbose=0)
# print("Baseline Error %.2f%%" % (100-scores[1]*100))


# In[31]:


# FOR CNN
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


# In[16]:


# def base_cnn():
#     model = Sequential()
#     model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(class_count, activation='softmax'))
#     # Compile
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model


# In[17]:


# cnn_model = base_cnn()
# cnn_model.fit(X_train, y_train, epochs=100, batch_size=200, verbose=2)


# In[32]:


def larger_cnn():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='linear'))
    model.add(LeakyReLU(alpha=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='linear'))
    model.add(LeakyReLU(alpha=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(50, activation='linear'))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(class_count, activation='softmax'))
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


cnn_model_larger = larger_cnn()
cnn_model_larger.fit(X_train, y_train, epochs=400, batch_size=200, verbose=2)


# In[21]:


# cnn_pred = cnn_model.predict(X_test)


# In[24]:


# cnn_pred_nums = np.array([np.argmax(pred) for pred in cnn_pred])


# In[28]:


# cnn_pred_final = np.array([(i, x) for (i, x) in enumerate(cnn_pred_nums)])


# In[31]:


# np.savetxt('../data/test_y.csv', cnn_pred_final, delimiter=',', fmt='%d', header='Id,Label')


# In[56]:


cnn_large_pred = cnn_model_larger.predict(X_test)


# In[57]:


cnn_large_pred_nums = np.array([np.argmax(pred) for pred in cnn_large_pred])


# In[58]:


cnn_large_pred_final = np.array([(i, x) for (i, x) in enumerate(cnn_large_pred_nums)])


# In[59]:


np.savetxt('../data/test_y_largecnn_rawdata.csv', cnn_large_pred_final, delimiter=',', fmt='%d', header='Id,Label')

