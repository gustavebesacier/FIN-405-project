from Utils import compute_rolling_betas, compute_equal_weight_data, compute_value_weighted_data, compute_ew_from_legs_data, compute_vw_from_legs_data, SEP,  VERBOSE, RF_COL
from Grapher import plot_mean_std_sr, get_mean_std_sharpe, plot_from_lists

import numpy as np
import pandas as pd


def mom_add_legs(data, col_decile='decile'):
    """Adds the leg for building the stragegy on.
       Create a column 'leg' that is 1 if the decile is 7, 8 or 9, and -1 if decile is 0, 1, 2"""

    data['leg'] = np.nan
    data.loc[data[col_decile] <= 2, 'leg'] = -1
    data.loc[data[col_decile] >= 7, 'leg'] = 1

    # Drop the observations that are in none of the legs
    data_mom = data.dropna().copy()

    return data_mom

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

def iv_question_b_ew(data, verbose=VERBOSE): # OK NO TOUCH BB

    EW_iv_piv = compute_equal_weight_data(data, col_ret='ret', col_decile='decile')

    return EW_iv_piv

def iv_question_b_vw(data, verbose=VERBOSE): # OK NO TOUCH BB

    data['VW_weight'] = data.groupby(['date', 'decile'])['mcap'].transform(lambda x: x / x.sum())
    data['VW_ret_contrib'] = data['VW_weight'] * data['ret']

    VW_iv_piv = compute_value_weighted_data(data, col_ret='VW_ret_contrib', col_decile='decile').rename(columns={'VW_ret_contrib': 'ret'})


    return VW_iv_piv

def iv_question_c_ew_compare_legs(data):
    """Returns the performance of each leg for each month. EW portfolios."""

    EW_iv_piv_leg = compute_equal_weight_data(data, col_ret='ret', col_decile='leg')

    return EW_iv_piv_leg

def iv_question_c_ew_compare_legs_to_strat(data_piv, show_plot = True, verbose = VERBOSE):
    # First aggregate the performance of each leg --> monthly performance of the strategy
    IV_ew_perf = compute_ew_from_legs_data(data_piv)

    # Compute mean, std and Sharpe ratio
    mean = IV_ew_perf['EW_return'].mean() * 12
    std = IV_ew_perf['EW_return'].std() * np.sqrt(12)
    rf = IV_ew_perf[RF_COL].mean() * 12

    performance_iv_ew = {'mean': mean, 'std': std, 'sharpe': (mean - rf) / std, 'rf': rf, 'n': len(IV_ew_perf['EW_return'])}

    # if verbose:
    print("IV strategy based on equally weighted portfolios")
    print(" - Expected return:\t {:.4f}".format(mean))
    print(" - Standard deviation:\t {:.4f}".format(std))
    print(" - Sharpe ratio:\t {:.4f}".format((mean - rf)/ std))
    print(" - t-stat:\t\t {:.4f}".format(mean / (std / np.sqrt(len(IV_ew_perf['EW_return'])))))
    
    return IV_ew_perf, performance_iv_ew

def iv_question_c_vw_compare_legs(data):
    """Returns the performance of each leg for each month. VW portfolios."""

    VW_iv_piv_leg, VW_iv_piv_leg_weights = compute_vw_from_legs_data(data, col_ret='ret', col_leg='leg')

    return VW_iv_piv_leg, VW_iv_piv_leg_weights

def iv_question_c_vw_compare_legs_to_strat(data):

    IV_vw_perf, IV_vw_weights = compute_vw_from_legs_data(data, col_ret='ret', col_leg='leg', col_mcap='mcap')
    # print("IVVV")
    # print(IV_vw_perf)

    # Create col 'ret' that is the return of the strategy sum(return leg 1 - return leg -1)
    # IV_vw_perf['ret'] = IV_vw_perf['VW_ret'] * IV_vw_perf['leg']
    IV_vw_perf = IV_vw_perf.groupby('date').agg({
            RF_COL: 'first',
            'VW_ret': 'mean'}).reset_index().rename(columns={'VW_ret': 'ret'})
    
    # Compute mean, std and Sharpe ratio
    mean = IV_vw_perf['ret'].mean() * 12
    std = IV_vw_perf['ret'].std() * np.sqrt(12)
    rf = IV_vw_perf[RF_COL].mean() * 12

    performance_iv_ew = {'mean': mean, 'std': std, 'sharpe': (mean - rf) / std, 'rf': rf, 'n': len(IV_vw_perf['ret'])}

    print("IV strategy based on value weighted portfolios")
    print(" - Expected return:\t {:.4f}".format(mean))
    print(" - Standard deviation:\t {:.4f}".format(std))
    print(" - Sharpe ratio:\t {:.4f}".format((mean - rf)/ std))
    print(" - t-stat:\t\t {:.4f}".format(mean / (std / np.sqrt(len(IV_vw_perf['ret'])))))

    return IV_vw_perf, performance_iv_ew, IV_vw_weights


def run_iv_part5(data, question_a=True, question_b=True, question_c = True, show_plot = True, verbose=VERBOSE):
    """ Run all the part 5, about the Idiosyncratic volatility strategy."""

    returns_iv = dict()
    
    data = iv_prepare_data(data).copy()


    if question_a:
        print(f"{SEP}\nQuestion a")
        data = iv_compute_idio_vol(data).copy()

        if verbose:
            print("Data of the IV strategy:")
            print(data.head(5))
            print(data.shape)

        returns_iv['IV_question_a'] = data.copy(deep = True)
        
        # Anticipate the next questions and avoid issues: create the deciles
        data['decile'] = data.groupby("date")["IV"].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    if question_b:
        
        returns_qb = dict()

        print(f"{SEP}\nQuestion b")
        if not question_a:
            raise ValueError("You need to run question a before question b in the IV strategy.")
        

        EW_iv_piv = iv_question_b_ew(data, verbose=verbose)
        VW_iv_piv = iv_question_b_vw(data, verbose=verbose)

        if show_plot:
        # Plot the results for the 2 different weightings
            plot_mean_std_sr(EW_iv_piv, '5b', "EW_returns_IV")
            plot_mean_std_sr(VW_iv_piv, '5b', "VW_returns_IV")
        
        returns_qb['IV_question_b_EW_returns'] = EW_iv_piv.copy(deep = True)
        returns_qb['IV_question_b_VW_returns'] = VW_iv_piv.copy(deep = True)

        returns_iv['IV_question_b'] = returns_qb# .copy(deep = True)

    if question_c:
        """Construct the idiosyncratic volatility factor."""

        print(f"{SEP}\nQuestion c")

        returns_question_c = dict()

        if not question_a: # to make sure we have the correct data
            raise ValueError("You need to run question a before question c in the IV strategy.")

        # Add the legs to the data (leg for high IV and low IV)
        data = mom_add_legs(data)
        data = data.dropna().copy()

        ## Equally weighted strategy
        # Compare each leg performance
        EW_iv_piv_leg = iv_question_c_ew_compare_legs(data)
        returns_question_c['IV_question_c_EW_returns_data'] = EW_iv_piv_leg.copy(deep = True)


        # Compare the performance of the strategy with the performance of each leg
        IV_ew_perf, performance_iv_ew = iv_question_c_ew_compare_legs_to_strat(data, show_plot = show_plot, verbose = True)
        # >>> performance_iv_ew = {'mean': 0.14420259964496035, 'std': 0.15926215094897586, 'sharpe': 0.6502576906002339, 'rf': 0.040641161168853454, 'n': 211}
        if verbose:
            print("Data and performance of the IV strategy using EW portoflios.")
            # print(IV_ew_perf)
            print(performance_iv_ew)
        returns_question_c['IV_question_c_EW_long_short_data'] = IV_ew_perf.copy(deep = True)
        returns_question_c['IV_question_c_EW_long_short_perf'] = performance_iv_ew#.copy(deep = True)

        if show_plot:
            mean, std, sr = get_mean_std_sharpe(EW_iv_piv_leg.rename(columns={'leg': 'decile'}))
            mean.append(performance_iv_ew['mean']), std.append(performance_iv_ew['std']), sr.append(performance_iv_ew['sharpe'])
            plot = plot_from_lists(mean, std, sr, plot_color = 'blue')
            plot.suptitle(f'Average portolio annualized mean return, standard deviation and sharpe ratio (EW_IV_legs_strat)')
            plot.savefig(f"Figures/question_5c_plot_EW_IV_legs_strat")
            plot.show()
            print("Bar 0: leg -1; Bar 1: leg 1; Bar 2: Strategy")
        
        ## Value weighted strategy
        # Compare each leg's performance
        VW_iv_piv_leg, VW_iv_piv_leg_weights = iv_question_c_vw_compare_legs(data)
        returns_question_c['IV_question_c_VW_returns_data'] = VW_iv_piv_leg.copy(deep = True)
        returns_question_c['IV_question_c_VW_returns_weights'] = VW_iv_piv_leg_weights.copy(deep = True)

        if show_plot:
            plot_mean_std_sr(VW_iv_piv_leg.rename(columns={'leg': 'decile', 'VW_ret':'ret'}), '5c', "VW_returns_IV_legs")
            print("In the graph, leg '-1' corresponds to bar '0'; leg '1' is bar '1'.")

        # Compare the performance of the strategy with the performance of each leg
        IV_vw_perf, performance_iv_vw, IV_vw_weights = iv_question_c_vw_compare_legs_to_strat(data)
        returns_question_c['IV_question_c_VW_long_short_data'] = IV_vw_perf.copy(deep = True)
        returns_question_c['IV_question_c_VW_long_short_perf'] = performance_iv_vw#.copy(deep = True)

        if verbose:
            print("Data and performance of the IV strategy using EW portoflios.")
            # print(IV_vw_perf)
            print(performance_iv_vw)

        if show_plot:
            mean, std, sr = get_mean_std_sharpe(EW_iv_piv_leg.rename(columns={'leg': 'decile'}))
            mean.append(performance_iv_vw['mean']), std.append(performance_iv_vw['std']), sr.append(performance_iv_vw['sharpe'])
            plot = plot_from_lists(mean, std, sr, plot_color = 'blue')
            plot.suptitle(f'Average portolio annualized mean return, standard deviation and sharpe ratio (VW_IV_legs_strat)')
            plot.savefig(f"Figures/question_5c_plot_VW_IV_legs_strat")
            plot.show()
            print("Bar 0: leg -1; Bar 1: leg 1; Bar 2: Strategy")

        # Add the returns to the dict of returns
        returns_iv['IV_question_c'] = returns_question_c#.copy(deep = True)


    return returns_iv