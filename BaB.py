import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

from Data_handler import RF_COL
from Grapher import plot_mean_std_sr
from Utils import compute_rolling_betas, compute_equal_weight_data, compute_value_weighted_data, VERBOSE

SAVE_TABLES = True


def bab_prepare_data(data_betas):

    # Create deciles based on Beta value (code from PS5)
    data_betas["decile"] = data_betas.groupby("date")["beta"].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    # Compute the value weighted contribution of each stock 
    data_betas['VW_weight'] = data_betas.groupby(['date', 'decile'])['mcap'].transform(lambda x: x / x.sum())
    data_betas['VW_ret_contrib'] = data_betas['VW_weight'] * data_betas['ret']

    return data_betas

def bab_equally_weighted_portfolios(data_betas):

    EW_returns = compute_equal_weight_data(data_betas, col_ret='ret', col_decile='decile')

    return EW_returns

def bab_value_weighted_portfolios(data_betas):

    VW_returns = compute_value_weighted_data(data_betas, col_ret = 'VW_ret_contrib', col_decile = 'decile').rename(columns={'VW_ret_contrib': 'ret'})

    return VW_returns

# def bab_get_portfolio_weights(data):
#     "Computes the weights of the Betting-Against-Beta portfolio (code inspired from PS5)."
#     df = data.copy()
#     df['z'] = df.groupby('date')['beta'].rank()                     # Assign each beta a rank, for each month
#     df['z_mean'] = df.groupby('date')['z'].transform('mean')        # Calculate the monthly mean the rank
#     df['norm'] = np.abs(df['z']- df['z_mean'])                      # Compute abs distance of rank to mean rank
#     df['sum_norm'] = df.groupby('date')['norm'].transform("sum")    # Sum the distance
#     df['k'] = 2 / df['sum_norm']                                    # Compute the k

#     # Compute the BAB weights
#     # df['wH'] = df['k'] * np.maximum(0, df['z'] - df['z_mean'])
#     # df['wL'] = - df['k'] * np.minimum(0, df['z'] - df['z_mean'])

#     df['wH'] = df['k'] * (df['z']- df['z_mean']) * ((df['z']- df['z_mean'])>0) 
#     df['wL'] = -df['k'] * (df['z']- df['z_mean']) * ((df['z']- df['z_mean'])<0)


#     # Drop irrelevant columns
#     df = df.drop(columns=["z_mean", 'z', 'norm', 'sum_norm', 'k'])

#     # Compute the weighted betas
#     df['bH'] = df['wH'] * df['beta']
#     df['bL'] = df['wL'] * df['beta']

#     # Compute individual contribution
#     df['rH'] = df['wH'] * df['ret']
#     df['rL'] = df['wL'] * df['ret']

#     # Compute the individual excess returns of the portfolios H and L
#     df['rH_e'] = df['wH'] * (df['ret'] - df[RF_COL])
#     df['rL_e'] = df['wL'] * (df['ret'] - df[RF_COL]) # Check that crazy formula bby 😃  (en gros, c'est okay de faire weight * excess return au lieu de faire weight * excess return?)
    
#     # Compute the return and betas of the two portfolios for each period
#     df_ = df.groupby('date').agg({
#         'rH_e': 'sum',
#         'rL_e': 'sum',
#         'rH': 'sum',
#         'rL': 'sum',
#         'bH': 'sum',
#         'bL': 'sum',
#         'Rm_e': 'first',
#     }).reset_index()

#     # Finally create the BAB portfolio return
#     df_['rBAB'] = df_['rL_e'] / df_['bL'] - df_['rH_e'] / df_['bH']
#     print("TA MER EN STRING de gueer")
#     print(df_)
#     return df_, df[["permno", "date", "wH", "wL"]]

def bab_get_portfolio_weights(data):
    """Code from PS5"""
    # Weights
    data['z'] = data.groupby('date')['beta'].transform(lambda x: x.rank())
    data['z_'] = data['z']-data.groupby('date')['z'].transform('mean')
    data['k'] = np.abs(data['z_'])
    data['k'] = 2/data.groupby('date')['k'].transform('sum')
    data['w_H'] = data['k'] * data['z_'] * (data['z_']>0) 
    data['w_L'] = -data['k'] * data['z_'] * (data['z_']<0) 

    # Weighted returns and beta
    data['beta_H'] = data['w_H'] * data['beta']
    data['beta_L'] = data['w_L'] * data['beta']
    data['R_H'] = data['w_H'] * data['ret']
    data['R_L'] = data['w_L'] * data['ret']
    data['R_H_e'] = data['w_H'] * data['Rn_e']
    data['R_L_e'] = data['w_L'] * data['Rn_e']
    # BAB = data.groupby('date')[['R_H','R_L','R_H_e','R_L_e','beta_H','beta_L', 'Rm_e']].sum().reset_index()

    BAB = data.groupby('date').agg({
        'R_H': 'sum',
        'R_L': 'sum',
        'R_H_e': 'sum',
        'R_L_e': 'sum',
        'beta_H': 'sum',
        'beta_L': 'sum',
        'Rm_e': 'first',
        }).reset_index()
    #     df_ = df.groupby('date').agg({
#         'rH_e': 'sum',
#         'rL_e': 'sum',
#         'rH': 'sum',
#         'rL': 'sum',
#         'bH': 'sum',
#         'bL': 'sum',
#         'Rm_e': 'first',
#     }).reset_index()

    # Levered and unlevered returns
    BAB['BAB1'] = BAB['R_L'] - BAB['R_H']
    BAB['rBAB'] = BAB['R_L_e']/BAB['beta_L'] - BAB['R_H_e']/BAB['beta_H']
    print("oui oui afpiu bwrpg  wiub")
    print(BAB)

    return BAB, data[["permno", "date", "w_H", "w_L"]]

def bab_question_b(data_betas, verbose = VERBOSE):

    EW_returns = bab_equally_weighted_portfolios(data_betas)
    VW_returns = bab_value_weighted_portfolios(data_betas)

    if verbose: 
        print("BAB Equally weighted returns per month, for each decile:")
        print(EW_returns.head(15))
        print(EW_returns.shape)

        print("BAB Value weighted returns per month, for each decile:")
        print(VW_returns.head(15))
        print(VW_returns.shape)

    # Plot the results for the 2 different weightings
    # plot_mean_std_sr(EW_returns, '3b', "EW_returns_BAB")
    # plot_mean_std_sr(VW_returns, '3b', "VW_returns_BAB")

    return EW_returns, VW_returns

def bab_question_cd(data_betas, verbose = VERBOSE):

    # Create the weights rBAB
    data_BAB, weights_BAB = bab_get_portfolio_weights(data_betas)

    return data_BAB, weights_BAB

def run_bab_part3(data, question_a=True, question_b = True, question_cd=True, show_plot = True, save_tables = True, verbose = VERBOSE):
    """ Run all the part 3, about the Betting-Against-Beta strategy."""

    returns = dict()

    if not os.path.exists("Tables"):
        os.makedirs("Tables")

    if question_a:
        data = compute_rolling_betas(data)
        returns['BAB_question_a_data'] = data.copy(deep = True)

    if question_b:
        
        returns_qb = dict()
        
        data = bab_prepare_data(data)
        EW_returns, VW_returns = bab_question_b(data, verbose = verbose)

        if show_plot:
            plot_mean_std_sr(EW_returns, '3b', "EW_returns_BAB")
            plot_mean_std_sr(VW_returns, '3b', "VW_returns_BAB")

        returns_qb['BAB_EW_returns_data'] = EW_returns.copy(deep = True)
        returns_qb['BAB_VW_returns_data'] = VW_returns.copy(deep = True)


        if save_tables:
            EW_returns.to_csv("Tables/3_BAB_qb_EW_return.csv", sep = ";")
            VW_returns.to_csv("Tables/3_BAB_qb_VW_return.csv", sep = ";")

        returns['BAB_question_b'] = returns_qb

    if question_cd:
        
        returns_qc = dict()

        bab_strategy, weights_BAB = bab_question_cd(data, verbose = verbose)

        returns_qc['BAB_qc_weights'] = weights_BAB.copy(deep = True)
        returns_qc['BAB_qc_strategy_data'] = bab_strategy.copy(deep = True)

        if verbose: 
            print("Data BAB portfolio:")
            print(bab_strategy.head(15))
            print(bab_strategy.shape)

        if save_tables:
            bab_strategy.to_csv("Tables/3_BAB_qcd.csv", sep = ";")
        
        # We compute the rf based on question b) results, as the underlying data is the same
        rf = np.mean(list(map(lambda x: 12*x, VW_returns.groupby('decile')[RF_COL].mean().values.tolist())))

        # Compute the return, std and Sharpe ratio of the BAB strategy
        BAB_ret = bab_strategy.rBAB.mean() * 12
        BAB_std = bab_strategy.rBAB.std() * np.sqrt(12)
        BAB_shr = (BAB_ret - rf) / BAB_std

        # Compute the CAPM alpha
        bab_strategy['one'] = 1 # Create the column for the constant
        model = sm.OLS(bab_strategy['rBAB'], bab_strategy[['one', 'Rm_e']]).fit() # Fit CAPM

        if verbose: 
            print("\n-----------------------\nBetting-against-beta strategy")
            print(" - Mean return: {:.2f}".format(BAB_ret))
            print(" - Standard deviation: {:.2f}".format(BAB_std))
            print(" - Sharpe ratio: {:.2f}".format(BAB_shr))
            print(" - CAPM alpha: {:.2f}".format(model.params.iloc[0] * 12))

        performances_bab = {'mean': BAB_ret, 'std': BAB_std, 'sharpe': BAB_shr, 'alpha': model.params.iloc[0] * 12, 'rf': rf}
        returns_qc['BAB_qc_strategy_perf'] = performances_bab
        
        
        returns['BAB_question_cd'] = returns_qc

    return returns