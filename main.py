import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
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

loaded = load_dataset('dataset')
dataset = loaded[0].reshape(len(loaded[0]), 10, 10, 1)
labels = np_utils.to_categorical(loaded[1], 4)

print('dataset shape:', dataset.shape)
print('labels shape:', labels.shape)

model = Sequential()
model.add(Conv2D(10,
        kernel_size=(2, 2),
        activation='relu',
        padding='same',
        input_shape=(10, 10, 1)))
model.add(MaxPooling2D(pool_size=(2, 2),
        strides=2,
        padding='same',
        data_format=None))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4, activation='relu'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(dataset, labels,
    batch_size=5, epochs=1000, verbose=1)
