import pandas as pd
import numpy as np
import statsmodels.api as sm
from termcolor import colored

from Data_handler import RF_COL
from Grapher import plot_mean_std_sr
from Utils import compute_rolling_betas

VERBOSE = False


def bab_prepare_data(data_betas):

    # Create deciles based on Beta value (code from PS5)
    data_betas["decile"] = data_betas.groupby("date")["beta"].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    # Compute the value weighted contribution of each stock 
    data_betas['VW_weight'] = data_betas.groupby(['date', 'decile'])['mcap'].transform(lambda x: x / x.sum())
    data_betas['VW_ret_contrib'] = data_betas['VW_weight'] * data_betas['ret']

    return data_betas

def bab_equally_weighted_portfolios(data_betas):

    # # Create deciles based on Beta value (code from PS5)
    # data_betas["EW_monthly_decile"] = data_betas.groupby("date")["beta"].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    # Equally weighted returns per month, for each decile
    EW_returns = data_betas.groupby(["date", "decile"]).agg({
        'ret': 'mean',
        RF_COL: 'first',
        'decile': 'first',
        'date': 'first'
        }).reset_index(drop=True)

    # print("Equally weighted returns per month, for each decile:")
    # print(EW_returns.head(5))

    return EW_returns

def bab_value_weighted_portfolios(data_betas):

    # data_betas['VW_weight'] = data_betas.groupby(['date', 'EW_monthly_decile'])['mcap'].transform(lambda x: x / x.sum())
    # data_betas['VW_ret_contrib'] = data_betas['VW_weight'] * data_betas['ret']

    VW_returns = data_betas.groupby(["date", "decile"]).agg({
        'VW_ret_contrib': 'sum',
        RF_COL: 'first',
        'decile': 'first',
        'date': 'first'
        }).reset_index(drop=True).rename(columns={'VW_ret_contrib': 'ret'})

    # print("Value weighted returns per month, for each decile:")
    # print(VW_returns.head(5))

    return VW_returns

def bab_get_portfolio_weights(data):
    """Computes the weights of the Betting-Against-Beta portfolio (code inspired from PS5)."""
    df = data.copy()
    df['z'] = df.groupby('date')['beta'].rank()                     # Assign each beta a rank, for each month
    df['z_mean'] = df.groupby('date')['z'].transform('mean')        # Calculate the monthly mean the rank
    df['norm'] = np.abs(df['z']- df['z_mean'])                      # Compute abs distance of rank to mean rank
    df['sum_norm'] = df.groupby('date')['norm'].transform("sum")    # Sum the distance
    df['k'] = 2 / df['sum_norm']                                    # Compute the k

    # Compute the BAB weights
    df['wH'] = df['k'] * np.maximum(0, df['z'] - df['z_mean'])
    df['wL'] = - df['k'] * np.minimum(0, df['z'] - df['z_mean'])

    # Drop irrelevant columns
    df = df.drop(columns=["z_mean", 'z', 'norm', 'sum_norm', 'k'])

    # Compute the weighted betas
    df['bH'] = df['wH'] * df['beta']
    df['bL'] = df['wL'] * df['beta']

    # Compute the individual excess returns of the portfolios H and L
    df['rH_e'] = df['wH'] * (df['ret'] - df[RF_COL])
    df['rL_e'] = df['wL'] * (df['ret'] - df[RF_COL]) # Check that crazy formula bby 😃  (en gros, c'est okay de faire weight * excess return au lieu de faire weight * excess return?)
    
    # Compute the return and betas of the two portfolios for each period
    df_ = df.groupby('date').agg({
        'rH_e': 'sum',
        'rL_e': 'sum',
        'bH': 'sum',
        'bL': 'sum',
        'Rm_e': 'first',
    }).reset_index()

    # Finally create the BAB portfolio return
    df_['rBAB'] = df_['rL_e'] / df_['bL'] - df_['rH_e'] / df_['bH']

    return df_

def bab_question_b(data_betas, verbose = VERBOSE):

    EW_returns = bab_equally_weighted_portfolios(data_betas)
    VW_returns = bab_value_weighted_portfolios(data_betas)

    if verbose: 
        print("Equally weighted returns per month, for each decile:")
        print(EW_returns.head(15))
        print(EW_returns.shape)

        print("Value weighted returns per month, for each decile:")
        print(VW_returns.head(15))
        print(VW_returns.shape)

    # Plot the results for the 2 different weightings
    plot_mean_std_sr(EW_returns, '3b', "EW_returns_BAB")
    plot_mean_std_sr(VW_returns, '3b', "VW_returns_BAB")

    return EW_returns, VW_returns

def bab_question_cd(data_betas, verbose = VERBOSE):

    # Create the weights rBAB
    data_BAB = bab_get_portfolio_weights(data_betas)

    if verbose:
        print("Data BAB portfolio:")
        print(data_BAB.head(15))
        print(data_BAB.shape)

    return data_BAB

def run_bab_part3(data, question_a=True, question_b = True, question_cd=True, verbose = VERBOSE):
    """ Run all the part 3, about the Betting-Against-Beta strategy."""

    if question_a:
        data = compute_rolling_betas(data)

    if question_b:
        data = bab_prepare_data(data)
        EW_returns, VW_returns = bab_question_b(data, verbose = True)


    if question_cd:
        bab_strategy = bab_question_cd(data, verbose = True)
        
        # We compute the rf based on question b) results, as the underlying data is the same
        rf = np.mean(list(map(lambda x: 12*x, VW_returns.groupby('decile')[RF_COL].mean().values.tolist())))

        # Compute the return, std and Sharpe ratio of the BAB strategy
        BAB_ret = bab_strategy.rBAB.mean() * 12
        BAB_std = bab_strategy.rBAB.std() * np.sqrt(12)
        BAB_shr = (BAB_ret - rf) / BAB_std

        # Compute the CAPM alpha
        bab_strategy['one'] = 1 # Create the column for the constant
        model = sm.OLS(bab_strategy['rBAB'], bab_strategy[['one', 'Rm_e']]).fit() # Fit CAPM

        print("\n-----------------------\nBetting-against-beta strategy")
        print(" - Mean return: {:.2f}".format(BAB_ret))
        print(" - Standard deviation: {:.2f}".format(BAB_std))
        print(" - Sharpe ratio: {:.2f}".format(BAB_shr))
        print(" - CAPM alpha: {:.2f}".format(model.params.iloc[0] * 12))
