import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, metrics
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from datetime import datetime

class BaseModel:
    def __init__(self, config_file, nameofrun, **kwargs):
        self.load_config(config_file)

        self.seed = self.config.get('seed')

        self.max_pooling = self.config['model']['max_pooling']
        self.reg = self.config['model']['regularization']
        self.lr = self.config['training']['learning_rate']
        self.epochs = self.config['training']['epochs']
        self.bach_size = self.config['training']['batch_size']
        self.use_early_stopping = self.config['training']['early_stop']['use']
        self.es_monitor = self.config['training']['early_stop']['monitor']
        self.patience  = self.config['training']['early_stop']['patience']
        self.classweights = self.config['training']['class_weights']
        self.window_size = self.config['data']['window_size']
        self.n_features = self.config['model']['n_features']
        self.min_epochs = self.config['training']['early_stop']['min_epochs']

        # overwrite evtl present arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.seed is not None:
            print(f"Setting seed to {self.seed}")
            tf.keras.utils.set_random_seed(self.seed)

        self.model = self.build_model()
        self.nameofrun = nameofrun



    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def build_model(self):
        raise NotImplementedError("Must override build_model")

    def compile_model(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        mdl_metrics = [
            metrics.CategoricalCrossentropy(name='NLL'), 
            metrics.CategoricalAccuracy(name='accuracy'), 
            metrics.Precision(name='precision'), 
            metrics.Recall(name='recall'), 
            metrics.F1Score(average='macro', name='f1_score')
        ]
        self.model.compile(loss=loss, optimizer=self.optimizer, metrics=mdl_metrics)

    def train_model(self, X_train, Y_train, X_valid, Y_valid):
        
        if self.classweights == 'balanced':
            Y_train_int = np.argmax(Y_train, axis=1)
            class_weights = dict(enumerate(compute_class_weight('balanced', classes = np.unique(Y_train_int), y = Y_train_int)))
        else:
            class_weights = None

        logdir = Path.cwd() / '..' / 'logs'/ self.config['training']['log_dir'] / (datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + self.nameofrun)    
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir= logdir, histogram_freq=self.config['training']['histogram_freq'])]
        if self.use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=self.es_monitor, 
                                                              patience=self.patientece, 
                                                              restore_best_weights=True,
                                                              start_from_epoch=self.min_epochs))


        self.model.fit(
            X_train, Y_train, 
            epochs=self.epochs, batch_size=self.bach_size, verbose=self.config['training']['verbose_settings'], shuffle=True, 
            callbacks=callbacks,
            validation_data=(X_valid, Y_valid), 
            class_weight=class_weights
        )