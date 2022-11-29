# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 15:43:37 2022

@author: Divyaj
"""

import tensorflow.compat.v1 as tf
tf.compat.v1.enable_eager_execution()
tf.config.experimental_run_functions_eagerly(True)

import sys, os
sys.path.append('D:\CS230_Deep_Learning\Projects\SimCLR_Defect_Detection')
import re
import numpy as np
import math

import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib
import matplotlib.pyplot as plt
from data_processing import preprocess_image
from keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers

from constants import *
import pathlib

data_labelled = pathlib.Path(labelled_data_path)
data_unlabelled = pathlib.Path(unlabelled_data_path)

image_count = len(list(data_labelled.glob('*/*.png'))) + \
    len(list(data_unlabelled.glob('*/*.png')))
print("Total number of images is %d" %(image_count))

print(INPUT_SHAPE)
"""
creating our simclr architecture based on the algorithm explained in :
https://sh-tsang.medium.com/review-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-5de42ba0bc66

Code credits: https://keras.io/examples/vision/semisupervised_simclr/

"""

#helper function for loading the data
def get_data(data_labelled, data_unlabelled, batch_size, IMG_SIZE):
    #loading labelled and unlabelled dataset together
    
    img_height, img_width = IMG_SIZE
    batch_size= batch_size
    unlabelled_train_ds = image_dataset_from_directory(
        data_unlabelled, 
        label_mode = None,
        seed=123,
        image_size = (img_height, img_width),
        batch_size = batch_size)
    
    labelled_train_ds = image_dataset_from_directory(
      data_labelled,
      validation_split=0.4,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    val_ds = image_dataset_from_directory(
      data_labelled,
      validation_split=0.4,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    """
    As the original dataset doesn't contain a test set, you will create one. 
    To do so, determine how many batches of data are available in the 
    validation set using tf.data.experimental.cardinality, 
    then move 20% of them to a test set.
    """

    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 5)
    val_ds = val_ds.skip(val_batches // 5)

    """
    Use buffered prefetching to load images from disk without having 
    I/O become blocking
    """
    AUTOTUNE = tf.data.AUTOTUNE

    #labelled_train_ds = labelled_train_ds.prefetch(buffer_size=AUTOTUNE)
    #unlabelled_train_ds = unlabelled_train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Labeled and unlabeled datasets are zipped together
    train_ds = tf.data.Dataset.zip(
        (unlabelled_train_ds, labelled_train_ds)
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, labelled_train_ds, test_ds


train_dataset, labelled_train_dataset, test_dataset = get_data(data_labelled,
                                                               data_unlabelled, 
                                                               batch_size, IMG_SIZE)

class Cut_Paste(layers.Layer):
    def __init__(self, x_scale = 10, y_scale = 10, IMG_SIZE = (224,224), **kwargs):
        super().__init__(**kwargs)
        
        """
        defining the x span and the y span of the box to cutout
        x_scale and y_scale are taken as inputs as % of the width and height of the image
        size
        """        
        self.size_x, self.size_y = IMG_SIZE
        self.span_x = int(x_scale*self.size_x*0.01)
        self.span_y = int(y_scale*self.size_y*0.01)
        
    #getting the vertices for cut and paste    
    def get_vertices(self):
        
        #determining random points for cut and paste 
        """ since the images in the dataset have the object of interest in the center of 
        the Image, the cutout will be taken from the central 25% of the image"""
        fraction = 0.25
        vert_x = random.randint(int(self.size_x*0.5*(1-fraction)),
                               int(self.size_x*0.5*(1+fraction)))        
        vert_y = random.randint(int(self.size_y*0.5*(1-fraction)),
                               int(self.size_y*0.5*(1+fraction)))
        
        start_x = int(vert_x-self.span_x/2)
        start_y = int(vert_y-self.span_y/2)
        end_x = int(vert_x+self.span_x/2)
        end_y = int(vert_y+self.span_y/2)
        
        return start_x, start_y, end_x, end_y
        
    def call(self, image):
        
        #getting random vertices for cutting
        cut_start_x, cut_start_y, cut_end_x, cut_end_y = self.get_vertices()
        
        #getting the image as a sub-image
        #image = tf.Variable(image)
        sub_image = image[cut_start_x:cut_end_x,cut_start_y:cut_end_y,:]
        
        #getting random vertices for pasting
        paste_start_x, paste_start_y, paste_end_x, paste_end_y = self.get_vertices()
        
        #replacing a part of the image at random location with sub_image
        image[paste_start_x:paste_end_x,
              paste_start_y:paste_end_y,:] = sub_image
        
        return tf.convert_to_tensor(image)
 
def get_encoder():
    base_model = keras.applications.resnet50.ResNet50(
        input_shape = INPUT_SHAPE,
        include_top = False, 
        weights='imagenet')
    encoder = keras.Sequential()
    encoder.add(base_model)
    encoder.add(layers.GlobalAveragePooling2D())
    return encoder


# Image augmentation module
def get_augmenter(min_area, brightness, jitter):
    zoom_factor = 1.0 - math.sqrt(min_area)
    return tf.keras.Sequential(
        [
            keras.Input(shape=(INPUT_SHAPE)),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
        ]
    )



class Contrastive_learning_model(keras.Model):
    def __init__(self):
        
        """
        super is used here since our defined class inherits from the class
        keras.Model. We don't have to rewrite the functions in keras.model again
        and super can diretly call it for us. This is called as supercharging a
        class"""
        super().__init__()
        
        self.temperature=temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.classification_augmenter = get_augmenter(**classification_augmentation)
        self.cut_paste = Cut_Paste(**cut_paste_augmentation)
        self.encoder = get_encoder()
        
        #Non-linear MLP as projection head
        self.projection_head = keras.Sequential([
                keras.Input(shape=(None,2,2048)),
                layers.Dense(width, activation="relu"),
                layers.Dense(width)], name="projection_head")
        
        # Single dense layer for linear probing
        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(None,1,2048)), layers.Dense(10)], name="linear_probe")
            
        self.encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()        
    
    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)
        
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer
        
        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")
            
    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy]
               
    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)
    
        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )
    
        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )
    
        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2       
            
    def train_step(self, data):
        (unlabeled_images), (labeled_images, labels) = data
        print(unlabeled_images.shape, labeled_images.shape)
        # Both labeled and unlabeled images are used, without labels
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)
        augmented_images_3 = self.cut_paste.call(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)
    
        # Labels are only used in evalutation for an on-the-fly logistic regression
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=True
        )
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
    
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        labeled_images, labels = data
    
        # For testing the components are used with a training=False flag
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
    
        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}
    
    
# Contrastive pretraining
pretraining_model = Contrastive_learning_model()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)

pretraining_history = pretraining_model.fit(
    train_dataset, epochs=num_epochs, validation_data=test_dataset
)
print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(pretraining_history.history["val_p_acc"]) * 100
    )
)

