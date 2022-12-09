import tensorflow.compat.v1 as tf
#tf.compat.v1.enable_eager_execution()
#tf.config.experimental_run_functions_eagerly(True)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from constants import *
import sys, os
import re
import numpy as np
import math


import matplotlib
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from mvtec_ad import mvtec_ad
from mvtec_ad_binary import mvtec_ad_binary

import pathlib
import random

from Contrastive_learning import prepare_dataset


# Load MVTEC dataset again
train_dataset, labeld_train_dataset, test_dataset = prepare_dataset(dataset_name_downstream)
print('dataset loaded.')



# Get pre-trained encoder model
model = keras.models.load_model("saved_model/encoder")
print('encoder model loaded SUCCESSFULLY.')

model.summary()

# Freeze all layers in the encoder
for layer in model.layers[:]:
  layer.trainable = False

"""
# Build downstream model for binary classification
model.add(layers.MaxPool2D((2,2), padding='valid'))
model.add(layers.GlobalAveragePooling2D('channels_last'))
model.add(layers.Dense(2)) # output 2 classes, good or anomaly
model.add(layers.Softmax(), name='downstream_model')
"""

downstream_model = keras.Sequential()
downstream_model.add(model)
#downstream_model.add(layers.MaxPool2D(pool_size=(2,2), padding="valid"))
#downstream_model.add(layers.GlobalAveragePooling1D())
downstream_model.add(layers.Dense(units=1024, activation="relu"))
downstream_model.add(layers.Dense(units=128, activation="relu"))
downstream_model.add(layers.Dense(units=2))
downstream_model.add(layers.Softmax())
                     
print('finished adding downstream task layers onto encoder model')

downstream_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                         loss=tf.keras.losses.BinaryCrossentropy(),
                         metrics=[tf.keras.metrics.BinaryAccuracy(),
                                  tf.keras.metrics.FalseNegatives()])

iterator = iter(train_dataset)
train_images, train_labels = next(iterator)
#print("train images:", tf.shape(train_dataset))
downstream_model.fit(train_images, train_labels, batch_size = 1, epochs=10, steps_per_epoch = 10)

iterator_test = iter(test_dataset)
test_images, test_labels = next(iterator_test)

loss, acc = downstream_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
            
