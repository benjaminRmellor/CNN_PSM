#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:47:16 2023

@author: benjaminmellor

"""

import xarray as xr
import numpy as np

def norm_extract(data, variable):
    
    """
    Function to extract mean and variance from variables for minmax 
    normalisation.
    
    Arguments: 
        variable: str, name of variable for normalisation
    Outputs:
        m: float, mean of variable
        v: float, varaince of variable
    """
    
    m = float(data[variable].mean())
    v = float(data[variable].var())

    return m, v

def normaliser(X, mu, sig):
    
    """"
    Function to normalise a variable.
    
    Arguments:
        X: float, variable
        mu: float, mean of variable
        sig: sig, variance of variable
    
    """
    return (X - mu)/sig

def znormalise_dataset(data, var_list):
    
    """
    Function to normalise variables across entire dataset
    
    Arguments:
        data: xarray dataset, input dataset
        var_list: list, variables to normalise
    Outputs:
        data: xarray dataset, out dataset
        norm_args_dict: dict of values for renormalising at later juncture

    """
    
    norm_args_dict = {}
    
    
    
    for var in var_list:
        # Extract mean
        m, v = norm_extract(data, var)
        
        # Store
        norm_args_dict[var]=[m, v]
        
        # Prep vars
        variables = [data[var], m, v]
        
        # Normalise
        data[var] = xr.apply_ufunc(normaliser,  # function to apply
                         *variables,  # pass arguments.
                         vectorize=True,
                         # vectorize the function: https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
                         dask='parallelized',  # let dask handle the parallelization
                         output_dtypes=[np.float64])  # data type of the output(s)

    return data, norm_args_dict

def train_val_test(df):
    
    """
    Function to split data into training, validation and testing datasets. 
    Converting to a numpy array for training. Training (65%), Testing (20%), 
    Validation (15%).
    
    Splits described in Lee et al., 2021: doi.org/10.1007/978-3-030-64777-3
    
    Arguments:
        df:: pandas dataframe, input dataset
    Outputs:
        train: pandas dataframe, train dataset
        val: pandas dataframe, validation dataset
        test: pandas dataframe, testing dataset
    
    """

    # Calculate the sizes of each split
    total_samples = len(df)
    train_size = int(0.65 * total_samples)
    test_size = int(0.2 * total_samples)
    validation_size = total_samples - train_size - test_size
    
    # Split the shuffled DataFrame into Train, Test, and Validation sets
    train_df = df[:train_size]
    test_df = df[train_size:train_size + test_size]
    validation_df = df[train_size + test_size:]
        
    return train_df, validation_df, test_df

def X_and_Y(data, Y_name):
    
    """
    Function to select X and Y variables for training
    
    Arguments:
        data: xarray dataset, input dataset
        Y_name: str, target Y variable
    Outputs:
        X: xarray dataset, X dataset
        Y: xarray dataset, Y dataset

    """
    
    Y = data[Y_name]
    
    X = data.drop_vars(Y_name)

    return X, Y

def out_pipeline(pred_a, out_n, base_da=None, var_n = None):
    
    A = pred_a

    if base_da:
        dims = base_da.dims
        coords = base_da.coords
        
        xrA = xr.DataArray(A,
                           dims = dims,
                           coords = coords)
        
        if var_n:
        
           xrA =  xrA.rename(var_n)
           
        
        xrA.to_netcdf(out_n)
        
    else:
        
        print('Dumping np.array as csv')
        
        A = np.array(A)
        
        np.savetxt(out_n, A, delimiter=",")
        
        
        
        
def pd_znormalise(df, index=0, cols=None):
    
    norm_args_dict = {}
        
    for n in cols:
        
        m = np.mean(df[n])
        sd = np.std(df[n])
        
        z = df[n]
        
        zn = (z - m) / sd
        
        df[f'{n}_zN'] = zn 

        # Store
        norm_args_dict[n]=[m, sd]
        
    return norm_args_dict   
        
# import xarray as xr
# import numpy as np
# import os 

# # Set dir
# os.chdir('/Users/benjaminmellor/GEOGM00XX_DISS')

# # Wd as string
# wd = os.getcwd()

# # Data store string
# dss = f'{wd}/cnn/data'

# # Extract data
# dat = xr.open_dataset(f'{dss}/era5_mclean.nc')

