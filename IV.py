from Utils import compute_rolling_betas, compute_equal_weight_data, compute_value_weighted_data, compute_ew_from_legs_data, compute_vw_from_legs_data, SEP,  VERBOSE, RF_COL
from Grapher import plot_mean_std_sr, get_mean_std_sharpe, plot_from_lists
from Momentum import mom_add_legs

import numpy as np
import pandas as pd

def iv_prepare_data(data):
    
    data = compute_rolling_betas(data)

    return data

def iv_compute_idio_vol(data_iv):
    """Computes the idiosyncratic volatility of the residuals against the CAPM model."""
    
    w = 60 # window size

    # Compute the residuals
    data_iv['residuals'] = data_iv['Rn_e'] - (data_iv['beta'] * data_iv['Rm_e'])

    # Compute volatility of residuals (idiosyncratic volatility)
    data_iv['IV'] = data_iv.groupby('permno')['residuals'].rolling(window=w, min_periods=36).std().reset_index(level=0, drop=True)
    data_iv = data_iv.dropna(subset=['IV']).copy()

    # Winsorize at 5% and 95% (code from PS5)
    data_iv['IV'] = data_iv['IV'].clip(data_iv['IV'].quantile(0.05), data_iv['IV'].quantile(0.95))

    data_iv = data_iv.drop(columns=['residuals']) # no need that column anymore

    return data_iv

# def iv_question_c_ew(data, verbose = VERBOSE):
    
#     # First, focus on the EW part
#     EW_iv_piv = compute_ew_from_legs_data(data, col_leg='leg', col_ret='ret')

#     # Compute mean, std and Sharpe ratio
#     mean = EW_iv_piv['EW_return'].mean() * 12
#     std = EW_iv_piv['EW_return'].std() * np.sqrt(12)
#     rf = EW_iv_piv[RF_COL].mean() * 12

#     if verbose:
#         print("IV strategy based on equally weighted portfolios")
#         print(" - Expected return:\t {:.2f}%".format(mean))
#         print(" - Standard deviation:\t {:.2f}%".format(std))
#         print(" - Sharpe ratio:\t {:.2f}".format((mean - rf)/ std))


#     performance_iv_ew = {'mean': mean, 'std': std, 'sharpe': (mean - rf) / std, 'rf': rf}

#     return EW_iv_piv, performance_iv_ew

# def iv_question_c_vw(data, verbose = VERBOSE):

#     VW_IV_piv = compute_vw_from_legs_data(data, col_ret='ret', col_leg='leg', col_mcap='mcap')

#     # Compute mean, std and Sharpe ratio
#     mean = VW_IV_piv['VW_ret'].mean() * 12
#     std = VW_IV_piv['VW_ret'].std() * np.sqrt(12)
#     rf = VW_IV_piv[RF_COL].mean() * 12

#     if verbose:
#         print("Momentum strategy based on value weighted portfolios")
#         print(" - Expected return:\t {:.2f}%".format(mean))
#         print(" - Standard deviation:\t {:.2f}%".format(std))
#         print(" - Sharpe ratio:\t {:.2f}".format((mean - rf)/ std))

#     performance_iv_vw = {'mean': mean, 'std': std, 'sharpe': (mean - rf) / std, 'rf': rf}

#     return VW_IV_piv, performance_iv_vw

def iv_question_b_ew(data, verbose=VERBOSE):

    EW_iv_piv = compute_equal_weight_data(data, col_ret='ret', col_decile='decile')

    return EW_iv_piv

def iv_question_b_vw(data, verbose=VERBOSE):

    data['VW_weight'] = data.groupby(['date', 'decile'])['mcap'].transform(lambda x: x / x.sum())
    data['VW_ret_contrib'] = data['VW_weight'] * data['ret']

    VW_iv_piv = compute_value_weighted_data(data, col_ret='VW_ret_contrib', col_decile='decile').rename(columns={'VW_ret_contrib': 'ret'})

    return VW_iv_piv

def iv_question_c_ew(data, show_plot = True):
        # Comparing performance of each leg
        EW_returns_IV_legs = data.groupby(["date", "leg"]).agg({
            'date': 'first',
            'ret': 'mean',
            RF_COL: 'first',
            'leg': 'first'
            }).reset_index(drop=True)
        
        if show_plot:
            plot_mean_std_sr(EW_returns_IV_legs.rename(columns={'leg': 'decile'}), '5c', "EW_returns_IV_legs")
            print("In the graph, leg '-1' corresponds to bar '0'; leg '1' is bar '1'.")

        # EW performance of going long the leg 1 and short leg -1
        EW_iv_piv = compute_ew_from_legs_data(data, col_ret='ret')

        # Compute mean, std and Sharpe ratio
        mean_EW_IV = EW_iv_piv['EW_return'].mean() * 12
        std_EW_IV = EW_iv_piv['EW_return'].std() * np.sqrt(12)
        rf_EW_IV = EW_iv_piv[RF_COL].mean() * 12
        n = len(EW_iv_piv['EW_return'])

        # Save the performances of the long-short strategy
        performance_iv_ew = {'mean': mean_EW_IV, 'std': std_EW_IV, 'sharpe': (mean_EW_IV - rf_EW_IV) / std_EW_IV, 'rf': rf_EW_IV, 'n': n}

        # Compare the performance from the two legs and the strategy
        m, v, s = get_mean_std_sharpe(EW_returns_IV_legs.rename(columns={'leg': 'decile'})) # recompute the values of the 2 legs
        m.append(mean_EW_IV), v.append(std_EW_IV), s.append((mean_EW_IV - rf_EW_IV) / std_EW_IV) # add to the lists the values for the portfolio
        if show_plot:
            plot = plot_from_lists(m, v, s, plot_color = 'blue')
            plot.suptitle(f'Average portolio annualized mean return, standard deviation and sharpe ratio (EW_IV_legs_strat)')
            plot.savefig(f"Figures/question_5c_plot_EW_IV_legs_strat")
            plot.show()
            print("Bar 0: leg -1; Bar 1: leg 1; Bar 2: Strategy")
        EW_returns_IV_legs

        return EW_iv_piv, performance_iv_ew


def iv_question_c_vw(data, verbose = VERBOSE, show_plot = True):
        
        VW_iv_piv = compute_vw_from_legs_data(data, col_ret='ret', col_leg='leg', col_mcap='mcap')

        if show_plot:
             plot_mean_std_sr(VW_iv_piv.rename(columns={'leg': 'decile', 'VW_ret':'ret'}), '5c', "VW_returns_IV_legs")
             print("In the graph, leg '-1' corresponds to bar '0'; leg '1' is bar '1'.")

        # Compute mean, std and Sharpe ratio
        mean_VW_IV = VW_iv_piv['VW_ret'].mean() * 12
        std_VW_IV = VW_iv_piv['VW_ret'].std() * np.sqrt(12)
        rf_VW_IV = VW_iv_piv[RF_COL].mean() * 12
        
        if verbose:
            print("IV strategy based on value weighted portfolios")
            print(" - Expected return:\t {:.2f}".format(mean_VW_IV))
            print(" - Standard deviation:\t {:.2f}".format(std_VW_IV))
            print(" - Sharpe ratio:\t {:.2f}".format((mean_VW_IV - rf_VW_IV)/ std_VW_IV))

        mean, std, sr = get_mean_std_sharpe(VW_iv_piv.rename(columns={'leg': 'decile', 'VW_ret':'ret'}))
        mean.append(mean_VW_IV), std.append(std_VW_IV), sr.append((mean_VW_IV - rf_VW_IV)/ std_VW_IV)

        if show_plot:
            plot = plot_from_lists(mean, std, sr, plot_color = 'blue')
            plot.suptitle(f'Average portolio annualized mean return, standard deviation and sharpe ratio (VW_IV_legs_strat)')
            plot.savefig(f"Figures/question_5c_plot_VW_IV_legs_strat")
            plot.show()
            print("Bar 0: leg -1; Bar 1: leg 1; Bar 2: Strategy")
        


def run_iv_part5(data, question_a=True, question_b=True, question_c = True, show_plot = True, verbose=VERBOSE):
    """ Run all the part 5, about the Idiosyncratic volatility strategy."""
    
    data = iv_prepare_data(data).copy()

    if question_a:
        print(f"{SEP}\nQuestion a")
        data = iv_compute_idio_vol(data).copy()
        if verbose:
            print(data.head(15))
            print(data.shape)

    if question_b:
        print(f"{SEP}\nQuestion b")
        if not question_a:
            raise ValueError("You need to run question a before question b in the IV strategy.")
        data['decile'] = data.groupby("date")["IV"].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

        EW_iv_piv = iv_question_b_ew(data, verbose=verbose)
        VW_iv_piv = iv_question_b_vw(data, verbose=verbose)

        if show_plot:
        # Plot the results for the 2 different weightings
            plot_mean_std_sr(EW_iv_piv, '5b', "EW_returns_IV")
            plot_mean_std_sr(VW_iv_piv, '5b', "VW_returns_IV")
        
    if question_c:

        data = mom_add_legs(data) # Add legs to the data to regroup the deciles
        data = data.dropna().copy()

        # Equally weighted strategy
        EW_piv_, EW_piv_perf = iv_question_c_ew(data, show_plot = False) # performances of the long-short strategy

        # Value weighted strategy
        iv_question_c_vw(data, show_plot=True)

        print(EW_piv_perf)


        # print(EW_piv_, EW_piv_perf)
        # EW_iv_perf = iv_question_c_ew(data, show_plot=show_plot)

        # Comparing performance of each leg

        # EW_returns_IV_legs = data.groupby(["date", "leg"]).agg({
        #     'date': 'first',
        #     'ret': 'mean',
        #     RF_COL: 'first',
        #     'leg': 'first'
        #     }).reset_index(drop=True)
        
        # plot_mean_std_sr(EW_returns_IV_legs.rename(columns={'leg': 'decile'}), '5c', "EW_returns_IV_legs")
        # print("In the graph, leg '-1' corresponds to bar '0'; leg '1' is bar '1'.")

        # # EW performance
        # EW_iv_piv, EW_iv_perf = iv_question_c_ew(data, verbose = True)

        # # For VW portfolios
        # VW_iv_piv, VW_iv_perf = iv_question_c_vw(data, verbose = True)

        # # Compare the long-short strategy with each leg's performance
        # mean, std, sr = get_mean_std_sharpe(EW_returns_IV_legs.rename(columns={'leg': 'decile'}))
        # mean.append(EW_iv_perf['mean']), std.append(EW_iv_perf['std']), sr.append((EW_iv_perf['mean'] - EW_iv_perf['rf'])/ EW_iv_perf['std'])

        # plot = plot_from_lists(mean, std, sr, plot_color = 'blue')

        # plot.suptitle(f'Average portolio annualized mean return, standard deviation and sharpe ratio (EW_IV_legs_strat)')
        # plot.savefig(f"Figures/question_5c_plot_EW_IV_legs_strat")
        # plot.show()
        # print("Bar 0: leg -1; Bar 1: leg 1; Bar 2: Strategy")


    return data