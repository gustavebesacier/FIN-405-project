from Utils import compute_equal_weight_data, compute_value_weighted_data, compute_ew_from_legs_data, compute_vw_from_legs_data, VERBOSE, RF_COL, SEP
from Grapher import plot_mean_std_sr

import pandas as pd
import numpy as np

def mom_prepare_data(data_mom):
    """Adds the rolling return column to the data, and create the deciles."""
    
    # Sort data by permno, then date
    data_mom.sort_values(by=['permno', 'date'], inplace=True)

    # Add a column for momentum return (last 12 months, excluding last month)
    data_mom['roll_ret'] = data_mom.groupby('permno').ret.transform(lambda x: x.rolling(11, closed='left').sum())

    # Create deciles for the momentum returns
    data_mom['decile'] = data_mom.groupby('date')['roll_ret'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    # Value weighted returns per month, for each decile
    data_mom['VW_weight'] = data_mom.groupby(['date', 'decile'])['mcap'].transform(lambda x: x / x.sum())
    data_mom['VW_ret_contrib'] = data_mom['VW_weight'] * data_mom['ret']

    return data_mom

def mom_equally_weighted_portfolios(data):
    """Returns the equally weighted returns per month, for each decile."""

    EW_returns = compute_equal_weight_data(data, col_ret='ret', col_decile='decile')

    return EW_returns

def mom_value_weighted_portfolios(data):

    """Returns the value weighted returns per month, for each decile."""

    VW_returns = compute_value_weighted_data(data, col_ret='VW_ret_contrib', col_decile='decile').rename(columns={'VW_ret_contrib': 'ret'})

    return VW_returns

def mom_add_legs(data, col_decile = 'decile'):
    """Adds the leg for building the stragegy on.
       Create a column 'leg' that is 1 if the decile is 7, 8 or 9, and -1 if decile is 0, 1, 2"""
    
    data['leg'] = np.nan
    data.loc[data[col_decile] <= 2, 'leg'] = -1
    data.loc[data[col_decile] >= 7, 'leg'] = 1

    # Drop the observations that are in none of the legs
    data_mom = data.dropna().copy()

    return data_mom


def mom_ew_from_legs(data):
    
    EW_mom_piv = compute_ew_from_legs_data(data, col_leg='leg', col_ret='ret')
    
    return EW_mom_piv


def mom_question_a(data, verbose = False, verbose_plus = VERBOSE):
    """Computes the EW and VW returns for each decile portfolio."""

    EW_returns = mom_equally_weighted_portfolios(data)
    VW_returns = mom_value_weighted_portfolios(data)


    return EW_returns, VW_returns

def mom_question_b_ew(data):
    
    # First, focus on the EW part
    EW_mom_piv = compute_ew_from_legs_data(data, col_leg='leg', col_ret='ret')

    # Compute mean, std and Sharpe ratio
    mean = EW_mom_piv['EW_return'].mean() * 12
    std = EW_mom_piv['EW_return'].std() * np.sqrt(12)
    rf = EW_mom_piv[RF_COL].mean() * 12

    performance_mom_ew = {'mean': mean, 'std': std, 'sharpe': (mean - rf) / std, 'rf': rf}

    return EW_mom_piv, performance_mom_ew

def mom_question_b_vw(data, verbose = VERBOSE):

    VW_mom_piv = compute_vw_from_legs_data(data, col_ret='ret', col_leg='leg', col_mcap='mcap')

    # Compute mean, std and Sharpe ratio
    mean = VW_mom_piv['VW_ret'].mean() * 12
    std = VW_mom_piv['VW_ret'].std() * np.sqrt(12)
    rf = VW_mom_piv[RF_COL].mean() * 12

    # if verbose:
    #     print("Momentum strategy based on value weighted portfolios")
    #     print(" - Expected return:\t {:.2f}%".format(mean))
    #     print(" - Standard deviation:\t {:.2f}%".format(std))
    #     print(" - Sharpe ratio:\t {:.2f}".format((mean - rf)/ std))

    performance_mom_vw = {'mean': mean, 'std': std, 'sharpe': (mean - rf) / std, 'rf': rf}

    return VW_mom_piv, performance_mom_vw


def run_mom_part4(data, question_a=True, question_b = True, save_tables = True, show_plot = True, verbose = VERBOSE):
    """ Run all the part 4, about the Momentum strategy."""

    returns = dict()

    data = mom_prepare_data(data).copy()

    returns['MOM_question_0_data'] = data.copy(deep = True)


    if question_a:
        returns_qa = dict()
        print(f"{SEP}\nQuestion a")
        EW_returns, VW_returns = mom_question_a(data, verbose = True)
        
        if save_tables:
            EW_returns.to_csv("Tables/4_MOM_qa_EW_return.csv", sep = ";")
            VW_returns.to_csv("Tables/4_MOM_qa_VW_return.csv", sep = ";")

        if show_plot:
            # Plot the results for the 2 different weightings
            plot_mean_std_sr(EW_returns, '4a', "EW_returns_MOM")
            plot_mean_std_sr(VW_returns, '4a', "VW_returns_MOM")

        returns_qa['MOM_qa_EW_returns_data'] = EW_returns.copy(deep = True)
        returns_qa['MOM_qa_VW_returns_data'] = VW_returns.copy(deep = True)

        returns['MOM_question_a'] = returns_qa

    if question_b:

        returns_qb = dict()

        print(f"{SEP}\nQuestion b")
        data = mom_add_legs(data)

        EW_mom_piv, EW_mom_perf = mom_question_b_ew(data)
        
        if verbose:
            print("Momentum strategy based on equally weighted portfolios")
            print(" - Expected return:\t {:.2f}%".format(EW_mom_perf['mean']))
            print(" - Standard deviation:\t {:.2f}%".format(EW_mom_perf['std']))
            print(" - Sharpe ratio:\t {:.2f}".format((EW_mom_perf['mean'] - EW_mom_perf['rf'])/ EW_mom_perf['std']))
        
        VW_mom_piv, VW_mom_perf = mom_question_b_vw(data, verbose = True)
        
        if verbose:
            print("Momentum strategy based on value weighted portfolios")
            print(" - Expected return:\t {:.2f}%".format(VW_mom_perf['mean']))
            print(" - Standard deviation:\t {:.2f}%".format(VW_mom_perf['std']))
            print(" - Sharpe ratio:\t {:.2f}".format((VW_mom_perf['mean'] - VW_mom_perf['rf'])/ VW_mom_perf['std']))

        if save_tables:
            EW_mom_piv.to_csv("Tables/4_MOM_qb_EW_return.csv", sep = ";")
            VW_mom_piv.to_csv("Tables/4_MOM_qb_VW_return.csv", sep = ";")


        returns_qb['MOM_qb_EW_returns'] = EW_mom_piv.copy(deep = True)
        returns_qb['MOM_qb_EW_performance'] = EW_mom_perf

        returns_qb['MOM_qb_VW_returns'] = VW_mom_piv.copy(deep = True)
        returns_qb['MOM_qb_VW_performance'] = VW_mom_perf

        returns['MOM_question_b'] = returns_qb

    return returns