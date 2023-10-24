#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:44:19 2023

@author: benjaminmellor
"""

import os 
import sys
import pandas as pd


# Set dir
os.chdir('/Users/benjaminmellor/GEOGM00XX_DISS')

# Wd as string
wd = os.getcwd()

sys.path.append('/Users/benjaminmellor/GEOGM00XX_DISS/cnn/scripts')
sys.path

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~ 1. Data Build and Clean ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import xarray as xr
import numpy as np
import data_utils as du

# Data store string
dss = f'{wd}/cnn/data'
dos = f'{wd}/cnn/outputs'
css = f'{wd}/cnn/configs'


def data_prep(k_dat = 'uk'):

    """
    Function to return the time series of a study site using the PRYSM model. 
    See Dee et al., (2015); Evans (2007).
    
    Inputs:
        n: name of study site
        data: xr datarray to be used to calculate and extract values
    
    Outputs:
        site_d18: Timeseries of predicted d18Otr 
    
    """

    # Data store string
    dss = f'{wd}/cnn/data'

    # Extract data
    dat = xr.open_dataset(f'{dss}/krig_limit_v2.nc')
    
    # Init list of variables needed to be normed
    var_list = ['t2m', 'tp', 'rh', 'ud18o', 'od18o']
    
    # Normalise data
    normed_data, nad = du.znormalise_dataset(dat, var_list)
    
    dat = dat
    
    # Using Universal Kriging
    if k_dat=='uk':
        
        dat = dat.drop('od18o')
        
        # Select X, Y
        X, y = du.X_and_Y(dat, 'ud18o')
        
    if k_dat=='ok':
        
        dat = dat.drop('ud18o')

        X, y = du.X_and_Y(dat, 'od18o')
        
    X = X.to_array()
    
    X = np.transpose(np.array(X), (1, 2, 3, 0))[:,1:,1:,:]
    y = np.array(y)[:,1:,:-1]
    
    from sklearn.model_selection import train_test_split

    # Assuming you have your feature matrix X and target vector y
    # X should be a 2D array-like structure, and y should be a 1D array-like structure

    # Split the data into a training set (usually 70-80% of the data) and a testing set (20-30% of the data)
    X_train, X_I, y_train, y_I = train_test_split(X, y, test_size=0.35, random_state=90210)
    
    X_test, X_val, y_test, y_val = train_test_split(X_I, y_I, test_size=.4, random_state=21252)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X, y


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~ MODEL REQUIREMENTS IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def to_ncdf(results, orig, fname, k_type='uk'):
    
    """
    Function to output the gCNN predictions. 
    See Dee et al., (2015); Evans (2007).
    
    Inputs:
        results: CNN predictions
        orig_ds: Original ncdf from which to extract dims etc
        fname: filename destination
    
    Outputs:
        fin_ds: TFinal xr Dataset of variables
    
    """
    
    ret = results
    
    if k_type == 'uk':
        d18o_mean = 24.57737138069013
        d18o_std = 20.051929706346915

    if k_type == 'ok':
        d18o_mean = 25.714772867302283
        d18o_std = 3.1147699667160547

    Uret = ret * d18o_std + d18o_mean

    years = orig.year

    gridy = orig.latitude[1:]
    gridx = orig.longitude[:-1]

    xr_Uret = xr.DataArray(data=Uret, 
                           coords={'year': years,'latitude': gridy, 'longitude': gridx}, 
                           dims=["year", "latitude", "longitude"])

    xr_Uret.to_netcdf(fname)


import gCNN as gn
import tensorflow as tf

from keras_tuner import BayesianOptimization

import gCNN as gn
    
if __name__ == '__main__':
    
    # Extract data
    dat = xr.open_dataset(f'{dss}/krig_limit_v2.nc')
    
    # Collect Data
    X_train, X_val, X_test, y_train, y_val, y_test, X, y  = data_prep('uk')

    
    CNN = gn.GlobalConvNet(input_data=X_train)
    
    callback = [tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss',
                      min_delta=0,
                      patience=5,
                      verbose=1, 
                      mode='auto',
    #                  start_from_epoch=250
                  )] 
    
    # Init Tuner
    tuner = BayesianOptimization(
        CNN.TunableBuildAlgos,
        objective='val_loss',
        max_trials=5,
        directory='/Volumes/BENSTON/DOCTR/tuner_output',
        project_name='algo_bayes'
        
            )

    # # Search space
    # tuner.search(X_train, 
    #              y_train, 
    #              batch_size=1, 
    #              epochs=100, 
    #              validation_data=(X_val, y_val)
    #              )
    
    # Select
    lmb = tuner.get_best_models(num_models=1)[0]
    
#     # # serialize model to JSON
#     # model_json = lmb.to_json()
    
#     # with open("/Volumes/BENSTON/DOCTR/gCNN_v1.json", "w") as json_file:
#     #     json_file.write(model_json)

#     # Save History
#     hista = lmb.fit(X_train, 
#             y_train,
#             batch_size=1,
# #            callbacks = callback,
#             validation_data=(X_val, y_val),
#             epochs = 500)
    
#     histb = lmb.fit(X_train,
#             y_train,
#             batch_size=1,
#             callbacks=callback,
#             validation_data=(X_val, y_val),
#             epochs = 500)


#     en = np.argmin(histb.history['val_loss'])
    
#     # Return output
#     y_hat = np.mean(lmb.predict(X), axis=1)

#     outname = f'/Volumes/BENSTON/DOCTR/gCNN_{500+en}_ret.nc'
    
#     to_ncdf(y_hat, dat, outname)
    
#     lmb.plot_history(histb, save=True, title='gCNN_3cE.png')


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEBUG SECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

