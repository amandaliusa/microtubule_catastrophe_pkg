import numpy as np
import pandas as pd

import bebi103 

"""
Functions to preprocessing microtubule catastrophe data.
"""

def process_labeled_unlabeled_data(filename):
    '''Processes csv file with microtubule catastrophe times 
    for labeled and unlabeled tubulin and returns tidy dataframe'''
    
    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)
    df['labeled'] = df['labeled'].apply(lambda x: 'labeled' if x else 'unlabeled')
    
    return df

def extract_labeled(df):
    '''Extracts microtubule catastrophe times for labeled tubulin 
    from provided dataframe'''
    
    labeled = df.loc[df['labeled']=='labeled']['time to catastrophe (s)']
    return labeled

def extract_labeled(df):
    '''Extracts microtubule catastrophe times for unlabeled tubulin 
    from provided dataframe'''
    
    unlabeled = df.loc[df['labeled']=='unlabeled']['time to catastrophe (s)']
    return unlabeled

def process_concentration_data(filename):
    '''Processes csv file with microtubule catastrophe times 
    for different tubulin concentrations'''
    
    df = pd.read_csv(filename, comment='#')
    df_tidy = pd.melt(df, var_name='concentration (uM)', value_name='catastrophe times (s)')
    df_tidy['concentration (uM)'] = df_tidy['concentration (uM)'].apply(lambda x: x[:-3])
    df_tidy = df_tidy.dropna()
    
    return df_tidy

def extract_concentration(df, conc):
    '''Extracts microtubule catastrophe times for specified 
    tubulin concentration'''
    
    return = df.loc[df['concentration (uM)'] == conc]['catastrophe times (s)']
