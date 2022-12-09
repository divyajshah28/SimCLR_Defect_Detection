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

import pathlib
import random

from Contrastive_learning import prepare_dataset


# Load MVTEC dataset again
train_dataset, labeld_train_dataset, test_dataset = prepare_dataset()
print('dataset loaded.')



# Get pre-trained encoder model
encoder = keras.models.load_model("saved_model/encoder")
print('encoder model loaded SUCCESSFULLY.')

# Freeze all layers in the encoder
for layer in encoder.layers[:]:
  layer.trainable = False


# Build downstream model for binary classification
encoder.add(layers.MaxPool2D((2,2), padding='valid'))
encoder.add(layers.GlobalAveragePooling2D('channels_last'))
encoder.add(layers.Dense(2)) # output 2 classes, good or anomaly
encoder.add(layers.Softmax()], name='downstream_model')

print('finished adding downstream task layers onto encoder model')



