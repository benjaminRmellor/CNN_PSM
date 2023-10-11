#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:00:55 2023

@author: benjaminmellor

ANN initialisation, insipred by Fang & Li (2019)  

https://doi.org/10.1029/2018MS001525

"""

import numpy as np
import tensorflow as tf
import os

cwd = os.getcwd()

# Layers Import
from keras.models import Sequential
from keras.layers import Dense, Reshape


def limit_mem():
    """By default TF uses all available GPU memory. This function prevents this."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)


# Build model Here
class ArtNeuNet():
    def __init__(self, input_data, stochastic=False, limit_mem=False, batch_size=32):
        
        """
        Initialisation of Artificial Neural Net. Stores data and hyperparameters.
        
        Arguments:
            data(np array): Contains all data used, T, Rh, P.
            
        """
        
        print("Initialising Model...")
        
        # Extract data shape
        self.input_shape = np.shape(input_data)
        
        self.length = self.input_shape[0]
        
        self.channels = self.input_shape[1]
        
        if stochastic:
            self.batch_size = 1
            
        else:
            self.batch_size = batch_size
            
            
        if limit_mem:
            limit_mem()

        print('Initialised...')
        
    def SetBatchSize(self, batch_size):
        
        self.batch_size = batch_size
        
    def TunableBuild(self, hp):
        
        print("Building Tuning Model...")

        model = Sequential()
        
        # Perform learning
        model.add(Dense(
            units  = 8,
            input_shape=(self.input_shape[1],),
            activation='relu'))
        
        # Output
        model.add(Dense(1))
        
        hp_learning_rate = hp.Float('learning_rate', 
                                    min_value=1e-5, 
                                    max_value=1e-1, 
                                    sampling='log')  
    
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp_learning_rate),
            loss='mean_squared_error',
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )
        
        return model
        
    def Build(self):
        
        print("Building Model...")

        model = Sequential()
        
        # Perform learning
        model.add(Dense(8, 
                        input_shape=(self.input_shape[1],),
                        activation='relu'))

        # Output Layer
        model.add(Dense((1)))

        # Learning Rate extracted from Weber et al., (2020)
        opt = tf.keras.optimizers.Adam(learning_rate = 0.06907177473013913)

        # Compile the model
        model.compile(
                    optimizer = opt,
                    loss='mse',
                    metrics=[tf.keras.metrics.RootMeanSquaredError()],
                        )
        
        self.model=model
        
    def Fit(self, X_train, Y_train, verbose=True, persist=False, epochs=100, **args):
        
        print("Fitting to training data...")
        
        #tf.debugging.assert_shapes([(X_train, self.input_shape)])
        
        #tf.debugging.assert_shapes([(Y_train, self.outshape)])
        
        print(f"\nTraining data is of shape{Y_train.shape}")
        
        
        
        callback = [tf.keras.callbacks.EarlyStopping(
                      monitor='loss',
                      min_delta=0,
                      patience=2,
                      verbose=1,
                      mode='auto'
                  )] 
        
        if verbose and persist==False:
        
            self.model.fit(
                x=X_train,
                y=Y_train,
                epochs = epochs,
                batch_size=self.batch_size,
                callbacks=callback,
                **args
                )
            
        if persist:
            self.model.fit(
                x=X_train,
                y=Y_train,
                epochs = epochs,
                batch_size=self.batch_size,
                **args
                )
            
        else:
            
            self.model.fit(x=X_train,
                           y=Y_train,
                           epochs=epochs
                           )
        
        print("Trained...")
        
    def Predict(self, X):
        
        self.results = self.model.predict(
            x=X,
            batch_size=np.shape(X)[0]
            )
        
        return self.results
    
    def Evaluate(self, X_val, Y_val):
        
        self.eval_results = self.model.evaluate(
            x=X_val,
            y=Y_val, 
            batch_size=1)
        
        return self.eval_results
    
    def LoadModel(self, location, set_def=False):
        
        from tensorflow.keras.models import model_from_json
        
        # load json and create model
        json_file = open(location, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # Compile the model
        loaded_model.compile(
                loss='mse',
                metrics=[tf.keras.metrics.RootMeanSquaredError()],
                    )
        
        
        if set_def:
            self.model = loaded_model
        
        return loaded_model
    
    def plot_history(hist, save=False, title=None):
        
        import matplotlib.pyplot as plt
        
        dos = f'{os.getcwd}/outputs'
        
        fig, ax = plt.subplots()
        
        ax.plot(hist.history['val_root_mean_squared_error'], label = 'val_RMSE', c='r', linewidth=.5)
        ax.plot(hist.history['loss'], label = 'loss', c='k', linewidth=.5)
        ax.plot(hist.history['root_mean_squared_error'], label='RMSE', c='b', linewidth=.5)
        
        ax.legend()
        ax.set_ylabel('Error')
        ax.set_xlabel('Epoch')

        if save:
            plt.savefig(f'{dos}/{title}.png')