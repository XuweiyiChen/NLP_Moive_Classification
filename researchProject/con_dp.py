import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import DataProcessing

train_samples, train_labels = DataProcessing.train_samples, DataProcessing.train_lables
test_samples, test_labels = DataProcessing.test_samples, DataProcessing.test_lables

print('Training set', train_samples[0].shape, train_labels[0].shape)
print('    Test set', test_samples[0].shape, test_labels[0].shape)

image_size = DataProcessing.image_size
num_labels = DataProcessing.num_labels
num_channels = DataProcessing.num_channels

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                           input_shape=[64, 64, 1], activation='relu'),
    tf.keras.layers.Conv2D(16, (3, 3)),
    tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                           activation='relu'),
    tf.keras.layers.Conv2D(16, (3, 3)),
    tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

tensorboard = TensorBoard(log_dir="drive/MyDrive/researchProject/researchProject/logs/{}".format(time()))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_samples, train_labels, batch_size=32, epochs=100, validation_data=(test_samples, test_labels), shuffle=True,
          callbacks=[tensorboard])

test_loss, test_acc = model.evaluate(test_samples, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# model.save('test_model.h5')