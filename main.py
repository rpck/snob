import numpy as np
import os
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
np.set_printoptions(threshold=np.inf)

def label_to_int(lbl):
    if lbl == 'line':
        return 0
    elif lbl == 'rectangle':
        return 1
    elif lbl == 'ellipse':
        return 2
    elif lbl == 'triangle':
        return 3
    return 0

def load_dataset(dataset_dir):
    # Our classifications
    dataset = []
    labels = []
    for subdir, dirs, files in os.walk(dataset_dir):
        for f in files:
            filepath = subdir + os.sep + f
            if filepath.endswith('.csv'):
                data = np.loadtxt(filepath, delimiter=',')
                # Get the subdirectory after the path seperator
                label = subdir[subdir.find(os.sep) + 1:]
                dataset.append(data)
                labels.append(label_to_int(label))
    return (np.array(dataset), np.array(labels))

# Load the dataset and reshape
loaded = load_dataset('dataset')
dataset = loaded[0].reshape(len(loaded[0]), 10, 10, 1)
labels = np_utils.to_categorical(loaded[1], 4)

print('dataset shape:', dataset.shape)
print('labels shape:', labels.shape)

# Use tanh instead of ReLU to prevent NaN errors
model = Sequential()
model.add(Conv2D(10,
        kernel_size=(2, 2),
        activation='tanh',
        padding='same',
        input_shape=(10, 10, 1)))
model.add(MaxPooling2D(pool_size=(2, 2),
        strides=2,
        padding='same',
        data_format=None))
model.add(Dropout(0.25))
model.add(Conv2D(10,
        kernel_size=(2, 2),
        activation='tanh',
        padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),
        strides=2,
        padding='same',
        data_format=None))
model.add(Flatten())
model.add(Dense(4, activation='relu'))

model.summary()

# Use a Stochastic-Gradient-Descent as a learning optimizer
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# Prevent kernel biases from being exactly 0 and giving nan errors
def constrainedCrossEntropy(ytrue, ypred):
    ypred = K.clip(ypred, 1e-7, 1e7)
    return losses.categorical_crossentropy(ytrue, ypred)

model.compile(loss=constrainedCrossEntropy, #'categorical_crossentropy',
    optimizer=sgd, #'adam',
    metrics=['accuracy'])

model.fit(dataset, labels,
    batch_size=16, epochs=1000, verbose=1)
