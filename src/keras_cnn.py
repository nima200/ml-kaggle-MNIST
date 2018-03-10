import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from pre_processing import *
K.set_image_dim_ordering('th')

#training, test = get_splits()
training = get_training()
test = get_test()
X_train = np.array([x for x, y in training])
X_test = np.array([x for x, y in test])
y_train = np.array([y for x, y in training])
y_test = np.array([y for x, y in test])

pixel_count = X_train.shape[1] * X_train.shape[2]

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Convert vectorized back to normal
y_train = np.array([np.argmax(x) for x in y_train])
y_train_vec = np_utils.to_categorical(y_train)
y_test_vec = np_utils.to_categorical(y_test)
class_count = y_test_vec.shape[1]

seed = 7
np.random.seed(seed)


X_train = X_train.reshape(X_train.shape[0], pixel_count).astype('float32')
X_test = X_test.reshape(X_test.shape[0], pixel_count).astype('float32')


def base_nn():
    nn_model = Sequential()
    # Relu hidden layer, same number of neurons as input layer
    nn_model.add(Dense(pixel_count, input_dim=pixel_count, kernel_initializer='normal', activation='relu'))
    # Softmax output layer
    nn_model.add(Dense(class_count, kernel_initializer='normal', activation='softmax'))
    # Compile
    nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nn_model


# # Create a base neural network model
# model = base_nn()
# model.fit(X_train, y_train_vec, validation_data=(X_test, y_test_vec), epochs=10, batch_size=50, verbose=2)
# scores = model.evaluate(X_test, y_test_vec, verbose=0)
# print("Baseline Error %.2f%%" % (100-scores[1]*100))


# FOR CNN
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


def base_cnn():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(class_count, activation='softmax'))
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


cnn_model = base_cnn()
cnn_model.fit(X_train, y_train_vec, validation_data=(X_test, y_test_vec), epochs=100, batch_size=200, verbose=2)
scores = cnn_model.evaluate(X_test, y_test_vec, verbose=0)
print("Baseline Error %.2f%%" % (100-scores[1]*100))
