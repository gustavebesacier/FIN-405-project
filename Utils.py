from Data_handler import RF_COL
from Grapher import plot_mean_std_sr

from termcolor import colored

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

VERBOSE = False
SEP = "------------------------"


def compute_rolling_betas(data, window_size = 60):

    covariance = data.set_index('date').groupby('permno')[['Rn_e', 'Rm_e']].rolling(window=window_size, min_periods=36).cov()
    betas = covariance.iloc[1::2,1].droplevel(2) / covariance.iloc[0::2,1].droplevel(2)
    betas = betas.dropna().reset_index().rename(columns={'Rm_e': 'beta'})

    # Make sure the dates columns are datetime
    betas.date = pd.to_datetime(betas.date)
    data.date = pd.to_datetime(data.date)

    # Offset the dates of the betas by 1 month (code from PS5)
    betas.date = betas.date + pd.DateOffset(months=1)

    # Merge the full data with betas (code from PS5)
    data_betas = pd.merge(data, betas, on=['permno', 'date'], how='left')


    # Finally, we winsorize the betas (5% and 95%) (code from PS5)
    data_betas['beta'] = data_betas['beta'].clip(data_betas['beta'].quantile(0.05), data_betas['beta'].quantile(0.95))

    # Drop all nan
    data_betas = data_betas.dropna().copy()


    return data_betas

def compute_equal_weight_data(data, col_ret = 'ret', col_decile = 'decile'):
    """Takes a dataframe as an input, specify the decile column and return columns. It returns a smaller dataframe of the expected return per decile per date based on equally weighted portfolios."""
    
    EW_returns = data.groupby(['date', col_decile]).agg({
        col_ret: 'mean',
        RF_COL: 'first',
        col_decile: 'first',
        'date': 'first'
        }).reset_index(drop=True)
    
    return EW_returns

def compute_value_weighted_data(data, col_ret = 'ret', col_decile = 'decile'):
    VW_returns = data.groupby(["date", col_decile]).agg({
        col_ret: 'sum',
        RF_COL: 'first',
        col_decile: 'first',
        'date': 'first'
        }).reset_index(drop=True)#.rename(columns={'VW_ret_contrib': 'ret'})
    
    return VW_returns

def compute_ew_from_legs_data(data, col_ret = 'ret', col_leg = 'leg'):
    """Returns a dataframe that computes the return of equally weighted portfolios of the legs."""
    EW_data = data.groupby(['date', col_leg]).agg({
        col_ret: 'mean', 
        RF_COL: 'first',
        col_leg: 'first',
        }).reset_index()

    EW_data_piv = EW_data.pivot(index='date', columns=col_leg, values='ret') # Pivot the data
    EW_data_piv['EW_return'] = EW_data_piv[1] - EW_data_piv[-1] # Compute the return of the EW momentum strategy as being the difference between the two legs
    EW_data_piv[RF_COL] = EW_data.groupby('date')[RF_COL].first() # Add the risk free rate
    EW_data_piv = EW_data_piv[['EW_return', RF_COL]]   # Keep only the relevant columns

    return EW_data_piv

def compute_vw_from_legs_data(data, col_ret = 'ret', col_leg = 'leg', col_mcap = 'mcap'):
    """Uses a data with legs. Compute the indiviudal contribution of each leg, and returns a dataframe of the perf."""
    VW_data_mom = data.copy()

    VW_data_mom['VW_wL'] = (VW_data_mom[col_leg] == -1) * VW_data_mom[col_mcap]
    VW_data_mom['VW_wL_sum'] = VW_data_mom.groupby('date')['VW_wL'].transform('sum')
    VW_data_mom['VW_wH'] = (VW_data_mom[col_leg] == 1) * VW_data_mom[col_mcap]
    VW_data_mom['VW_wH_sum'] = VW_data_mom.groupby('date')['VW_wH'].transform('sum')
    VW_data_mom['VW_wL'] = VW_data_mom['VW_wL'] / VW_data_mom['VW_wL_sum']
    VW_data_mom['VW_wH'] = VW_data_mom['VW_wH'] / VW_data_mom['VW_wH_sum']
    VW_data_mom = VW_data_mom.drop(columns=['VW_wL_sum', 'VW_wH_sum'])
    VW_data_mom['VW_w'] = VW_data_mom['VW_wL'] * VW_data_mom[col_leg] + VW_data_mom['VW_wH'] * VW_data_mom[col_leg]
    VW_data_mom['VW_ret'] = VW_data_mom['VW_w'] * VW_data_mom[col_ret]

    # Create a dataframe that aggregates the returns, at each month and keep the risk free rate
    VW_data_mom_ = VW_data_mom.groupby(['date']).agg({
        'VW_ret': 'sum', 
        RF_COL: 'first',
        }).reset_index()
    
    return VW_data_mom_