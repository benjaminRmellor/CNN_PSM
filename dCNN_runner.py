#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 22:31:09 2023

@author: benjaminmellor


The purpose of this script is an example of the CNNa utilised in the dissertation
study titled: 
    
    "Improving Tree Ring D18O Estimations Using a Convolutional 
                        Neural Netowrk Approach"
                        
Here we demonstrate how the chosen method, CNNa, is developed and initialised.
Further to this, hyperparemeters can be found in the CNN used, calling the .summary().
As this is the final tuned version. It is not pretrained, so should be retrained 
as per the methodology set out in the work. Of a 250 epoch spin-up before the
early-stopping function is applied. 

"""

import os 
import sys
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf

# Wd as string
wd = os.getcwd()
dss = f'{wd}/data'
css = f'{wd}/configs'
sod = f'{wd}/outputs'

from utils import data_utils as du
from scripts import dCNN

def predictand_ts_cnn(name, year, data, loc_dat):
    
    """
    Function to return the time series of a study site using the PRYSM model. 
    See Dee et al., (2015); Evans (2007).
    
    Inputs:
        n: name of study site
        data: xr datarray to be used to calculate and extract values
    
    Outputs:
        site_d18: Timeseries of predicted d18Otr 
    
    """
    
    lat0 = loc_dat.loc[name]['Lat']
    lon0 = loc_dat.loc[name]['Lon']
    
    lat_bounds = [lat0+1.625, lat0-1.625]
    lon_bounds = [lon0+1.625, lon0-1.625]

    dat_ = data.sel(latitude = slice(lat_bounds[0], lat_bounds[1]))
    dat_ = dat_.sel(longitude = slice(lon_bounds[1], lon_bounds[0]))
    dat_ = dat_.sel(year=year)
                    
    o = np.array(dat_.to_array())

    return o

def plot_history(hist, save=False, title=None):

    import matplotlib.pyplot as plt

    dos = sod

    fig, ax = plt.subplots()

    ax.plot(hist.history['val_loss'], label = 'val_MSE', c='b', linewidth=.5)
    ax.plot(hist.history['loss'], label = 'MSE', c='r', linewidth=.5)
    


    ax.legend()
    ax.set_ylabel('Error')
    ax.set_xlabel('Epoch')

    if save:
        plt.savefig(f'{dos}/{title}.png')
        

def data_prep():

    z_dat = pd.read_excel(f'{dss}/obs_ts_1940.xlsx')
    loc_dat = pd.read_excel(f'{dss}/tr_meta.xlsx')
    loc_dat.set_index('Series Name', inplace=True)
    z_dat.set_index('Year', inplace=True)
    
    z_dat = z_dat.drop('baker_ecu', axis=1)
    
    dat_str = f'{dss}/era5_wrh.nc'

    dat = xr.open_dataset(dat_str)
    dat = dat.sel(year=slice(1940, 2010))

    dat['t2m_zN'] = (dat.t2m - dat.t2m.mean()) / dat.t2m.std()
    dat['tp_zN'] = (dat.tp - dat.tp.mean()) / dat.tp.std()
    dat['rh_zN'] = (dat.rh - dat.rh.mean()) / dat.rh.std()
    
    ndat = dat[['t2m_zN', 'tp_zN', 'rh_zN']]

    # Stack the DataFrame and remove rows with all NaN values
    z_stack = z_dat.stack().dropna()

    data_list = []


    for i, meta in enumerate(z_stack.index):
    
        year = meta[0]
        name = meta[1]
        
        z = z_stack.loc[meta]
        
        lat = loc_dat.loc[name].Lat
        lon = loc_dat.loc[name].Lon
    
        o = predictand_ts_cnn(name, year, data=ndat, loc_dat=loc_dat)
        
        o = o.reshape((13, 13, 3))
    
            # Append the data to the list
        data_list.append({
        "year": year,
        "name": name,
        "lat": lat,
        "lon": lon,
        "X": o,
        "d18o": z
    })

    # Create a DataFrame from the list
    result_df = pd.DataFrame(data_list)
    
    result_df['d18o_zN'] = (result_df.d18o - result_df.d18o.mean()) / result_df.d18o.std()
    
    # Shuffle the DataFrame
    shuffled_df = result_df.sample(frac=1, random_state=90210) 
    

    return shuffled_df

if __name__ == '__main__':
    
    df = data_prep()
    
    # Unnest the arrays
    df_unnested = df.explode("X", ignore_index=True)

    # Convert the "X" column to a NumPy array and reshape
    unnested_arrays = np.stack(df_unnested["X"].values)
    reshaped_arrays = unnested_arrays.reshape((-1, 13, 13, 3))
    
    X = reshaped_arrays
    
    X_wh = np.array(X)

    # Split Df
    df_Train, df_Val, df_Test = du.train_val_test(df)
    
    # Set Xs    
    X_tr, X_va, X_ts = du.train_val_test(X)
    
    # Set Yx
    y_tr = np.array(df_Train.d18o_zN)
    y_va = np.array(df_Val.d18o_zN)
    y_ts = np.array(df_Test.d18o_zN)
    
    # Init dCNN
    CNN = dCNN.ConvNet(input_data=X_tr)
    
    # Callbacks to be used if desired
    callback = [tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss',
                      min_delta=0,
                      patience=5,
                      verbose=1, 
                      mode='auto'
                  )] 

    # Load model
    lm = CNN.LoadModel(f'{wd}/configs/dCNN.json')
    
    # Show summary
    lm.summary()
    
    # Fit
    lm.fit(X_tr, y_tr, epochs=1000)
    
    # Predict all events
    y_hat = np.mean(lm.predict(X_wh), axis=1)
    
    # Unnomrmalise
    df['d18o_hat'] = y_hat * df.d18o.std() + df.d18o.mean()
    
    
    # If save unhash
    # df.to_csv(f'{sod}/dCNN_1000E.csv')

