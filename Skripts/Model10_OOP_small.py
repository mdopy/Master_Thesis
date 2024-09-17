import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, metrics
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from datetime import datetime
import importlib
import BaseModel
importlib.reload(BaseModel)
from BaseModel import BaseModel

class ReshapeAndPadLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReshapeAndPadLayer, self).__init__(**kwargs)

    def call(self, inputs):
        shape = tf.shape(inputs)
        # Reshape
        X_angles = tf.reshape(inputs[:, :, :20], (shape[0], shape[1], 5, 4))
        # Pad
        X_angles = tf.pad(X_angles, [[0, 0], [2, 0], [0, 0], [0, 0]], 'CONSTANT')
        # Transpose
        X_angles = tf.transpose(X_angles, perm=[0, 3, 1, 2])
        # Extract point quaternion part
        X_pointquat = inputs[:, :, 20:27]
        return {'X_angles': X_angles, 'X_pointquat': X_pointquat}

    def compute_output_shape(self, input_shape):
        X_angles_shape = (input_shape[0], 4, input_shape[1] + 2, 5)
        X_pointquat_shape = (input_shape[0], input_shape[1], 7)
        return {'X_angles': X_angles_shape, 'X_pointquat': X_pointquat_shape}



class ConvModel(BaseModel):
    def build_model(self):


        handgestwindow = Input(shape=[self.window_size, self.n_features])
        x = ReshapeAndPadLayer()(handgestwindow)

        x1 = layers.Conv2D(filters=6, kernel_size=(4, 3), padding='valid', kernel_regularizer=l2(self.reg), activation='relu')(x['X_angles'])
        x1 = layers.Reshape((self.config['data']['window_size'], -1))(x1)
        x1 = layers.Conv1D(filters=16, kernel_size=3, padding='valid', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu')(x1)
        x1 = layers.Conv1D(filters=64, kernel_size=3, padding='valid', dilation_rate=2, kernel_regularizer=l2(self.reg), activation='relu')(x1)
        x1 = layers.Conv1D(filters=64, kernel_size=3, padding='valid', dilation_rate=4, kernel_regularizer=l2(self.reg), activation='relu')(x1)
        # if self.max_pooling:
        #     x1 = layers.MaxPool1D(pool_size=3, strides=None, padding='valid')(x1)

        x2 = layers.Conv1D(filters=16, kernel_size=3, padding='valid', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu')(x['X_pointquat'])
        x2 = layers.Conv1D(filters=64, kernel_size=3, padding='valid', dilation_rate=2, kernel_regularizer=l2(self.reg), activation='relu')(x2)
        x2 = layers.Conv1D(filters=64, kernel_size=3, padding='valid', dilation_rate=4, kernel_regularizer=l2(self.reg), activation='relu')(x2)
        # if self.max_pooling:
        #     x2 = layers.MaxPool1D(pool_size=3, strides=None, padding='valid')(x2)

        concat = layers.concatenate([x1, x2])
        if self.max_pooling:
            x = layers.MaxPool1D(pool_size=3, strides=None, padding='valid')(concat)
            x = layers.Flatten()(x)
        else: 
            x = layers.Flatten()(concat)

        x = layers.Dense(200, kernel_regularizer=l2(self.reg), activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        output_layer = layers.Dense(self.config['data']['n_classes'], activation='softmax')(x)

        model = Model(inputs=handgestwindow, outputs=output_layer)
        return model
    



