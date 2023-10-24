#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 19:09:29 2023

@author: benjaminmellor

The purpose of this script is an example of the CNN utilised in the dissertation
study titled: 
    
    "Improving Tree Ring D18O Estimations Using a Convolutional 
                        Neural Netowrk Approach"
                        
Here we demonstrae how the chosen method, a CNN, is developed and initialised. In
absence of the dsatasets used within the study, we have generated random data 
for placeholders. Further to this, various hyperparemeters such as feature number, 
learning rate etc, have been given placeholder values that reflect the likely
ultimate outcome of tuning we are performing. 



                        
                        

"""

import numpy as np
import tensorflow as tf

# Layers Import
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, BatchNormalization, Dropout, RandomRotation


def limit_mem():
    """By default TF uses all available GPU memory. This function prevents this."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

# Build model Here
class GlobalConvNet():
    def __init__(self, input_data, save = False):
        
        """
        Initialisation of Global Convolution Net. Stores data and hyperparameters.
        
        Arguments:
            data(np array): Contains all data used, T, Rh, P.
            save(boolean): If true, saves model checkpoints. Default false. 
            
        """
        
        print("Initialising Model...")
        
        # Extract data shape
        self.input_shape = np.shape(input_data)
        
        
        self.dt = self.input_shape[0]
        self.dx = self.input_shape[1]
        self.dy = self.input_shape[2]
        self.layer_n = self.input_shape[3]
        
        self.gridsize = self.dx*self.dy
        
        self.outshape = self.dt*self.gridsize
        
        self.fin_dense_units = self.dt*self.dx*self.dy
        
        print('Initialised...')
        
    def Build(self):
        
        print("Building Model...")

        print(f"Input Shape: {self.input_shape}")
        print(f"Output Shape: {self.outshape}")
        

        print(f'FDU{self.fin_dense_units}')

    # Class to build CNN
        model = Sequential()
        
    # Add the first 3D convolutional layer
        model.add(Conv2D(filters = 24, 
                         kernel_size=(3, 3), 
                         activation='relu',  
                         input_shape = self.input_shape[1:]))
        
    # Batch normalise
        model.add(BatchNormalization(axis = -1))

    # Add a second 2D convolutional layer
        model.add(Conv2D(filters = 48, 
                          kernel_size=(3, 3), 
                          activation='relu'))
        
    # Batch normalise
        model.add(BatchNormalization(axis = -1))

    # Add another max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # Add a second 2D convolutional layer
        model.add(Conv2D(filters = 48, 
                              kernel_size=(3, 3), 
                              activation='relu'))
            
    # Batch normalise
        model.add(BatchNormalization(axis = -1))

    # Add another max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # Dropout Possibility
        model.add(Dropout(rate=.5))        
        
    # Flatten the output from the previous layer
        model.add(Flatten())

    # Fully connected layer 64 units
        model.add(Dense(128, activation='softmax'))

    # Dropout
        model.add(Dropout(rate=.5))
        
    # Final Dense
        model.add(Dense(64, activation='softmax'))

    # Turn into grid
        model.add(Dense(self.n_samples, activation = 'linear'))
    
    # Learning Rate extracted from Weber et al., (2020)
        opt = tf.keras.optimizers.Adam(learning_rate = 0.06907177473013913)

    # Compile the model
        model.compile(
            optimizer = opt,
            loss='mse',
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
                )

   # Associate model
        self.model = model

        print("Model Built...")
        
    def TunableBuildArch(self, hp):
        
        print("Building Model...")

        print(f"Input Shape: {self.input_shape}")
        print(f"Output Shape: {self.outshape}")

        print(f'FDU{self.fin_dense_units}')
        
        unit_arr = [24, 32, 48, 64, 128, 244]
        
    # Class to build CNN
        model = Sequential()
        
    # Add the first 3D convolutional layer
        model.add(Conv2D(
            filters = hp.Int(
                name='Conv1',
                min_value = 32,
                max_value = 256,
                step=32
                ), 
            kernel_size = hp.Choice(name='Conv1_kernel',
                values=[3, 5]
                ),
                activation='relu',
                input_shape = self.input_shape[1:]
                )
            )
        
    # Batch normalise
        model.add(BatchNormalization(axis = -1))

    # Add a second 2D convolutional layer
        model.add(Conv2D(
            filters = hp.Int(
                name='Conv2',
                min_value = 32,
                max_value = 256,
                step=32
                ), 
            kernel_size=hp.Choice(
                name='Conv2_kernel',
                values=[3, 5]
                ), 
                activation='relu'
                )
            )
        
    # Batch normalise
        model.add(BatchNormalization(axis = -1))

    # Add another max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # Add a third 2D convolutional layer
        model.add(Conv2D(
            filters = hp.Int(
                name='Conv3',
                min_value = 32,
                max_value = 256,
                step=32
                ), 
                         kernel_size=hp.Choice(
                             name='Conv3_kernel',
                             values=[3, 5]
                                               ), 
                         activation='relu',  
                         )
            )
            
    # Batch normalise
        model.add(BatchNormalization(axis = -1))

    # Add another max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # Dropout Possibility
        model.add(Dropout(rate=.5))        
        
    # Flatten the output from the previous layer
        model.add(Flatten())

    # Fully connected layer 64 units
        model.add(Dense(
            units = hp.Int(
                name='Dense1',
                min_value = 32,
                max_value = 256,
                step=32
                ), 
            activation='softmax'))

    # Dropout
    #    model.add(Dropout(rate=))
        
    # Final Dense
        # Fully connected layer 64 units
        model.add(Dense(
            units = hp.Int(
            name='Dense1',
            min_value = 32,
            max_value = 256,
            step=32), 
            activation='softmax'))

    # Turn into grid
        model.add(Dense(self.fin_dense_units, activation = 'linear'))
        
        model.add(Reshape((self.dt, self.dx, self.dy)))
    
    # Learning Rate extracted from Weber et al., (2020)
        opt = tf.keras.optimizers.Adam(learning_rate = 0.06907177473013913)

    # Compile the model
        model.compile(
            optimizer = opt,
            loss='mse',
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
                )
        
        return model
        
        
        
        
    def TunableBuildAlgos(self, hp):
        
        print("Building Model...")

        print(f"Input Shape: {self.input_shape}")
        print(f"Output Shape: {self.outshape}")
        
        fin_dense_units = self.dx*self.dy
        print(f'FDU{fin_dense_units}')

    # Class to build CNN
        model = Sequential()
        
    # Add the first 3D convolutional layer
        model.add(Conv2D(filters = 128, 
                         kernel_size=(5, 5), 
                         activation='relu',  
                         input_shape = self.input_shape[1:]))
        
    # Batch normalise
        model.add(BatchNormalization(axis = -1))

    # Add a second 2D convolutional layer
        model.add(Conv2D(filters = 244, 
                          kernel_size=(5, 5), 
                          activation='relu'))
        
    # Batch normalise
        model.add(BatchNormalization(axis = -1))

    # Add another max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # Add a second 2D convolutional layer
        model.add(Conv2D(filters = 64, 
                              kernel_size=(3, 3), 
                              activation='relu'))
            
    # Batch normalise
        model.add(BatchNormalization(axis = -1))

    # Add another max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # Dropout Possibility
        model.add(Dropout(rate=hp.Float('drop1',
                                      max_value = .5,
                                      min_value = 0)))        
        
    # Flatten the output from the previous layer
        model.add(Flatten())

    # Fully connected layer 64 units
        model.add(Dense(units = 64, 
                        activation='softmax'))

    # Dropout
        model.add(Dropout(rate=hp.Float('drop2',
                                      max_value = .5,
                                      min_value = 0)))     
        
    # Final Dense
        model.add(Dense(64, activation='softmax'))

    # Turn into grid
        model.add(Dense(self.fin_dense_units, activation = 'linear'))
        
        model.add(Reshape((self.dt, self.dx, self.dy)))
    
    # Learning Rate extracted from Weber et al., (2020)
        opt = tf.keras.optimizers.Adam(learning_rate = hp.Float('lr',
                                                              min_value=0.001,
                                                              max_value=0.1,
                                                              sampling="log"))

    # Compile the model
        model.compile(
            optimizer = opt,
            loss='mse',
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
            )

   # Associate model
   
        return model
        
    def Fit(self, X_train, Y_train, batch_size=1, verbose=True, epochs=100, **args):
        
        print("Fitting to training data...")
        
        tf.debugging.assert_shapes([(X_train, self.input_shape)])
        
        tf.debugging.assert_shapes([(Y_train, self.outshape)])
        
        print(f"\nTraining data is of shape{Y_train.shape}")
        
        
        
        callback = [tf.keras.callbacks.EarlyStopping(
                      monitor='loss',
                      min_delta=0,
                      patience=2,
                      verbose=1, 
                      mode='auto'
                  )] 
        
        if verbose:
        
            self.model.fit(
                x=X_train,
                y=Y_train,
                batch_size=batch_size,
                epochs = epochs,
                callbacks=callback,
                **args
                )
            
            
        else:
            
            self.model.fit(x=X_train,
                           y=Y_train,
                           epochs=100
                           )
        
        print("Trained...")
        
    def Predict(self, X):
        
        self.results = self.model.predict(
            x=X
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
        
        dos = '/Users/benjaminmellor/GEOGM00XX_DISS/cnn/outputs'
        
        fig, ax = plt.subplots()
        
        ax.plot(hist.history['val_root_mean_squared_error'], label = 'val_RMSE', c='r', linewidth=.5)
        ax.plot(hist.history['loss'], label = 'loss', c='k', linewidth=.5)
        ax.plot(hist.history['root_mean_squared_error'], label='RMSE', c='b', linewidth=.5)
        
        ax.legend()
        ax.set_ylabel('Error')
        ax.set_xlabel('Epoch')

        if save:
            plt.savefig(f'{dos}/{title}.png')
        
    
limit_mem()
