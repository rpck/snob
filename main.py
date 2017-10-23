import numpy as np
import os
import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import re

batch_size = 16
epochs = 5000

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

def load_dataset(dataset_dir, parity):
    # Our classifications
    dataset = []
    labels = []
    for subdir, dirs, files in os.walk(dataset_dir):
        for f in files:
            filepath = subdir + os.sep + f
            if filepath.endswith('.csv') and int(re.findall('\d+', filepath)[0]) % 2 == parity:
                print(filepath)
                data = np.loadtxt(filepath, delimiter=',')
                # Get the subdirectory after the path seperator
                label = subdir[subdir.find(os.sep) + 1:]
                dataset.append(data)
                labels.append(label_to_int(label))
    return (np.array(dataset), np.array(labels))

# Load the dataset and reshape
train_set = load_dataset('dataset', 1)
test_set = load_dataset('dataset', 0)

input_shape = (1, 10, 10)
train_dataset = ()
test_dataset = ()
if K.image_data_format() == 'channels_first':
    train_dataset = train_set[0].reshape(len(train_set[0]), 1, 10, 10)
    test_dataset = test_set[0].reshape(len(test_set[0]), 1, 10, 10)
    input_shape = (1, 10, 10)
else:
    train_dataset = train_set[0].reshape(len(train_set[0]), 10, 10, 1)
    test_dataset = test_set[0].reshape(len(test_set[0]), 10, 10, 1)
    input_shape = (10, 10, 1)

train_labels = np_utils.to_categorical(train_set[1], 4)
test_labels = np_utils.to_categorical(test_set[1], 4)

print('dataset shape:', train_dataset.shape)
print('labels shape:', train_labels.shape)

# Use tanh instead of ReLU to prevent NaN errors
model = Sequential()
model.add(Conv2D(10,
        kernel_size=(2, 2),
        activation='tanh',
        padding='same',
        input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(1, 1),
        strides=2,
        padding='same',
        data_format=None))
model.add(Dropout(0.1))
model.add(Conv2D(10,
        kernel_size=(2, 2),
        activation='tanh',
        padding='same'))
model.add(MaxPooling2D(pool_size=(1, 1),
        strides=2,
        padding='same',
        data_format=None))
model.add(Flatten())

#"Squash" to probabilities
model.add(Dense(4, activation='softmax'))

model.summary()
# Use a Stochastic-Gradient-Descent as a learning optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Prevent kernel biases from being exactly 0 and giving nan errors
def constrainedCrossEntropy(ytrue, ypred):
    ypred = K.clip(ypred, 1e-7, 1e7)
    return losses.categorical_crossentropy(ytrue, ypred)
model.compile(loss=constrainedCrossEntropy,
              optimizer=sgd,
              metrics=['accuracy'])
filepath = 'model.h5'
model.fit(train_dataset, train_labels,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(test_dataset, test_labels))

#Evaluate on other half of dataset
score = model.evaluate(test_dataset, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Saves the model
#Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#Serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
