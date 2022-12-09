# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 20:33:32 2022

@author: Divyaj
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 15:43:37 2022

@author: Divyaj
"""

import tensorflow.compat.v1 as tf
tf.compat.v1.enable_eager_execution()
tf.config.experimental_run_functions_eagerly(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from constants import *
import sys, os
#sys.path.append(wdir)
import re
import numpy as np
import math

# import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib
import matplotlib.pyplot as plt
#from data_processing import preprocess_image
from keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from mvtec_ad import mvtec_ad

import pathlib
import random

"""
creating our simclr architecture based on the algorithm explained in :
https://sh-tsang.medium.com/review-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-5de42ba0bc66

Code credits: https://keras.io/examples/vision/semisupervised_simclr/

"""

def prepare_dataset(dataset_name):
    # Labeled and unlabeled samples are loaded synchronously
    # with batch sizes selected accordingly
    steps_per_epoch = (unlabeled_dataset_size + labeled_dataset_size) // batch_size
    unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch
    labeled_batch_size = labeled_dataset_size // steps_per_epoch
    print(
        f"batch size is {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)"
    )
    
    labeled_train_dataset = (
        tfds.load(dataset_name, split="train", as_supervised=True, shuffle_files=True)
    )

    def image_resize(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (96,96))
        return image, label

    labeled_train_dataset = labeled_train_dataset.map(image_resize)
    labeled_train_dataset = labeled_train_dataset.batch(labeled_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
   
    test_dataset = (
        tfds.load(dataset_name, split="test", as_supervised=True)
    )

    test_dataset = test_dataset.map(image_resize)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    train_dataset = labeled_train_dataset
    
    return train_dataset, labeled_train_dataset, test_dataset


# Load MVtec_AD dataset
train_dataset, labeld_train_dataset, test_dataset = prepare_dataset(dataset_name_downstream)

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
        image = tf.Variable(image).numpy()
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

"""
Our model calss inherits from the built in class called keras.Model. 
Our class is the sub-class and the keras.Model is the super class
-- also called as the parent class or the base class.

This way of creating a class from another class is also called as sub-classing
or inheritance"""

class Contrastive_learning_model(keras.Model):
    
    
    """ This article describes why self is explicitly defined in python 
    and why it is necessary. It basically helps us to create multiple objects
    by creating multiple instances. Every time a new object is called, self
    will create a new instance for that class
    
    Source: https://www.programiz.com/article/python-self-why#:~:text=If%20there%20was%20no%20self,its%20own%20attributes%20and%20methods.
    """
    
    def __init__(self):
        
        '''
        super is used here since our defined class inherits from the class
        keras.Model. We don't have to rewrite the functions in keras.model again
        and super can diretly call it for us. This is called as supercharging a
        class
        '''
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
            [layers.Input(shape=(None,1,2048)), 
             layers.Dense(10)], name="linear_probe")
            
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
               
    def cross_entropy(self, similarities):
        # similarities is a matrix of shape N*N where N is the batch size
        self.exp_n = np.exp(similarities)
        loss =  np.sum(np.divide(np.diagonal(self.exp_n), 
                                    np.sum(self.exp_n, axis=1)-np.diagonal(self.exp_n)))
    
        return tf.convert_to_tensor(loss)
    
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
            
    """the below function called train_step is called by the Model.fit method
    and the batch size and epoch iteration is taken care of internally"""
    
    def train_step(self, data):
        (images, labels) = data
        
        # Both labeled and unlabeled images are used, without labels
        #images = tf.concat((unlabeled_images, labeled_images), axis=0)
        #images = tf.images.resize(images, (224,224,3))
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images, training=False)
        augmented_images_2 = self.contrastive_augmenter(images, training=False)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=False)
            features_2 = self.encoder(augmented_images_2, training=False)
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
            images, training=False
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
    
if __name__ == '__main__':


    # Contrastive pretraining
    pretraining_model = Contrastive_learning_model()
    pretraining_model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),run_eagerly=True
        )

    """ the fit function of the keras.Model will call the train_step method.
    If the train_step method is not defined, it will pick the method from the
    parent calss. If defined, it will take the function from the sub-class 
    since the object is defined for the sub-class"""


    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1)

    pretraining_history = pretraining_model.fit(
        train_dataset, epochs=num_epochs, validation_data=test_dataset,
        callbacks=[cp_callback]
        )

    saved_model_path = pathlib.Path("./saved_model")
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)
    tf.keras.models.save_model(pretraining_model.encoder, "saved_model/encoder")
#    tf.saved_model.save(pretraining_model, "saved_model/pretraining_model")

    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(pretraining_history.history["val_p_acc"]) * 100
            )
        )

    loss, acc = pretraining_model.evaluate(test_dataset, verbose=2)
    print("Pretraining model, accuracy: {:5.2f}%".format(100 * acc))



"""
# Load MVTEC dataset again
train_dataset, labeld_train_dataset, test_dataset = prepare_dataset()

# Get pre-trained encoder model
encoder = pretraining_model.encoder

# Build downstream model for binary classification
downstream_model = keras.Sequential([
    keras.Input(shape=(None,2,2048)),
    layers.MaxPool2D((2,2), padding='valid'),
    layers.GlobalAveragePooling2D('channels_last'),
    layers.Dense(2), # output 2 classes, good or anomaly
    layers.Softmax()], name='downstream_model')


"""
