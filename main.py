import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
np.set_printoptions(threshold=np.inf)

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
                labels.append(label)
    return (dataset, labels)

loaded = load_dataset('dataset')
dataset = loaded[0]
labels = loaded[1]

model = Sequential()
model.add(Conv2D(10,
        kernel_size=(2, 2),
        activation='relu',
        padding='valid',
        input_shape=(60, 10, 10)))
model.add(MaxPooling2D(pool_size=(2, 2),
        strides=None,
        padding='valid',
        data_format=None))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2, activation='relu'))

model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(dataset, labels,
    batch_size=5, epochs=5, verbose=1)
