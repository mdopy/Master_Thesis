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
        resh = ReshapeAndPadLayer()(handgestwindow)

        x = layers.Conv2D(filters=16, kernel_size=(4,3), padding='valid', kernel_regularizer=l2(self.reg), activation='relu')(resh['X_angles'])
        x = layers.Reshape((self.window_size, -1))(x)
        
        concat = layers.concatenate([x, resh['X_pointquat']])
        x = layers.Conv1D(filters=32, kernel_size=3, padding='same', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu')(concat)
        x = layers.Conv1D(filters=32, kernel_size=3, padding='same', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu')(x)
        block1out = layers.MaxPool1D(pool_size=2, strides = None, padding='valid')(x)

        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu')(block1out)
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu')(x)
        block2out = layers.MaxPool1D(pool_size=2, strides = None, padding='valid')(x)

        x = layers.Conv1D(filters=128, kernel_size=3, padding='same', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu')(block2out)
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu')(x)
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu')(x)
        block3out = layers.MaxPool1D(pool_size=2, strides = None, padding='valid')(x)


        x = layers.Conv1D(filters=128, kernel_size=18, padding='same', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu', name = 'c_head1')(block3out)
        x = layers.Dropout(0.5)(x)
        x = layers.Conv1D(filters=128, kernel_size=1, padding='same', dilation_rate=1, kernel_regularizer=l2(self.reg), activation='relu', name = 'c_head2')(x)
        headout = layers.Dropout(0.5)(x)


        # shape number of channels to n classes
        stage1 = layers.Conv1D(filters=self.config['data']['n_classes'],kernel_size=1,padding="same",strides=1,activation="relu", name = 'c_red1')(headout)
        # Up-sample to original image size
        shape = stage1.shape
        stage1 = layers.Reshape((shape[1], 1, shape[2]))(stage1)
        stage1 = layers.UpSampling2D(size=(2,1),interpolation="bilinear")(stage1)
        shape = stage1.shape
        stage1 = layers.Reshape((shape[1], shape[3]))(stage1)

        
        # upsampleblock 2
        # shape number of channels to n classes
        stage2 = layers.Conv1D(filters=self.config['data']['n_classes'],kernel_size=1,padding="same",strides=1,activation="relu", name = 'c_red2')(block2out)


        # merge
        x = layers.Add()([stage1, stage2])

        # Get Softmax outputs for all classes
        # x = layers.Conv1D(filters=self.config['data']['n_classes'],kernel_size=1,activation="softmax",padding="same",strides=1)(x)
        shape = x.shape
        x = layers.Reshape((shape[1], 1, shape[2]))(x)
        # Up-sample to original image size
        x = layers.UpSampling2D(size=(4,1),interpolation="bilinear")(x)
        shape = x.shape
        x = layers.Reshape((shape[1], shape[3]))(x)
        output_layer = layers.Softmax(axis = -1)(x)
        model = Model(inputs=handgestwindow, outputs=output_layer)

        return model
    
    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        mdl_metrics = [
            metrics.CategoricalCrossentropy(name='NLL'), 
            metrics.CategoricalAccuracy(name='accuracy'), 
            metrics.Precision(name='precision'), 
            metrics.Recall(name='recall'), 
            # metrics.F1Score(average='macro', name='f1_score')
        ]
        self.model.compile(loss=loss, optimizer=optimizer, metrics=mdl_metrics)

    def train_model(self, X_train, Y_train, X_valid, Y_valid):
        # if classweights are ot used, its not necessary to reimolement this method
        if self.classweights == 'balanced':
            Y_train_int = np.argmax(np.reshape(Y_train, (-1, Y_train.shape[-1])), axis=1)
            class_weights = dict(enumerate(compute_class_weight('balanced', classes = np.unique(Y_train_int), y = Y_train_int)))
        else:
            class_weights = None

        logdir = Path.cwd() / '..' / 'logs'/ self.config['training']['log_dir'] / (datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + self.nameofrun)    
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir= logdir, histogram_freq=self.config['training']['histogram_freq'])]
        if self.use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=self.es_monitor, 
                                                              patience=self.patience, 
                                                              restore_best_weights=True))


        self.model.fit(
            X_train, Y_train, 
            epochs=self.epochs, batch_size=self.bach_size, verbose=self.config['training']['verbose_settings'], shuffle=True, 
            callbacks=callbacks,
            validation_data=(X_valid, Y_valid), 
            class_weight=class_weights
        )



