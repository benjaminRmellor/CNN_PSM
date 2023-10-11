#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:53:18 2023

@author: benjaminmellor
"""


import os 
import sys
import numpy as np
import pandas as pd

# Wd as string
wd = os.getcwd()
dss = f'{wd}/data'
css = f'{wd}/configs'
sod = f'{wd}/outputs'

from utils import data_utils as du
from scripts import fANN

# Data store string

def LoadData():
    
    # Extract data
    df = pd.read_csv(f'{dss}/SHUFF_ANNT.csv', index_col=0)

    # Init list of variables needed to be normed
    var_list = ['T', 'P','Rh', 'd18o']

    # Normalise data
    nad = du.pd_znormalise(df, cols = var_list)
    
    return nad, df

if __name__ == '__main__':
    
    # Load Shuffled Dataframe
    NormDict, df = LoadData()
    
    # Split data
    df_Train, df_Val, df_Test = du.train_val_test(df)
    
    # Select X, Y
    X_tr, X_v, X_te = df_Train[['T_zN', 'P_zN', 'Rh_zN']], df_Val[['T_zN', 'P_zN', 'Rh_zN']], df_Test[['T_zN', 'P_zN', 'Rh_zN']]
    y_tr, y_v, y_te = df_Train['d18o_zN'], df_Val['d18o_zN'], df_Val['d18o_zN']

    # Train
    X_tr = np.array(X_tr) 
    y_tr = np.array(y_tr)

    # Val
    X_v = np.array(X_v)
    X_y = np.array(y_v)

    # Ensemble datasets
    X_wh = np.array(df[['T_zN', 'P_zN', 'Rh_zN']])
    y_wh = np.array(df['d18o_zN'])
    
    # Initialise Neural Network
    ANN = fANN.ArtNeuNet(input_data = np.array(X_tr))
    
    # If desired, use example architecture:
    model = ANN.LoadModel(f'{css}/fANN.json')
    
    # This version is the optimal model selected by Keras Tuner. 
    
    # Train new init
    model.fit(X_tr, 
            y_tr,
            validation_data=(X_v, y_v),
            epochs = 10000 # as per Fang & Li, 2019
            )
    
    # Predict
    y_hat = model.predict(X_wh)
    
    # Unnormalise + Add to DF
    df['d18o_hat'] = y_hat * NormDict['d18o'][1] + NormDict['d18o'][0]

    # Output
    df.to_csv(f'{sod}/fANN_shuffled.csv')