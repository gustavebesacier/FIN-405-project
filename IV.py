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

    if verbose:
        print("IV strategy based on equally weighted portfolios")
        print(" - Expected return:\t {:.2f}%".format(mean))
        print(" - Standard deviation:\t {:.2f}%".format(std))
        print(" - Sharpe ratio:\t {:.2f}".format((mean - rf)/ std))

    performance_iv_ew = {'mean': mean, 'std': std, 'sharpe': (mean - rf) / std, 'rf': rf, 'n': len(IV_ew_perf['EW_return'])}

    if show_plot:
        # Plot the comparison of the performance of the strategy against the performance of the legs
        pass
    
    return IV_ew_perf, performance_iv_ew

def iv_question_c_vw_compare_legs(data):
    """Returns the performance of each leg for each month. VW portfolios."""

    VW_iv_piv_leg = compute_vw_from_legs_data(data, col_ret='ret', col_leg='leg')

    return VW_iv_piv_leg

def iv_question_c_vw_compare_legs_to_strat(data):

    IV_vw_perf = compute_vw_from_legs_data(data, col_ret='ret', col_leg='leg', col_mcap='mcap')
    print("IVVV")
    print(IV_vw_perf)

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

    return IV_vw_perf, performance_iv_ew


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
        
        # Anticipate the next questions and avoid issues: create the deciles
        data['decile'] = data.groupby("date")["IV"].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

    if question_b:
        print(f"{SEP}\nQuestion b")
        if not question_a:
            raise ValueError("You need to run question a before question b in the IV strategy.")
        

        EW_iv_piv = iv_question_b_ew(data, verbose=verbose)
        VW_iv_piv = iv_question_b_vw(data, verbose=verbose)

        if show_plot:
        # Plot the results for the 2 different weightings
            plot_mean_std_sr(EW_iv_piv, '5b', "EW_returns_IV")
            plot_mean_std_sr(VW_iv_piv, '5b', "VW_returns_IV")
        
    if question_c:
        """Construct the idiosyncratic volatility factor."""

        if not question_a: # to make sure we have the correct data
            raise ValueError("You need to run question a before question c in the IV strategy.")

        # Add the legs to the data (leg for high IV and low IV)
        data = mom_add_legs(data)
        data = data.dropna().copy()


        ## Equally weighted strategy
        # Compare each leg performance
        EW_iv_piv_leg = iv_question_c_ew_compare_legs(data)

        # Compare the performance of the strategy with the performance of each leg
        IV_ew_perf, performance_iv_ew = iv_question_c_ew_compare_legs_to_strat(data, show_plot = show_plot, verbose = True)
        # >>> performance_iv_ew = {'mean': 0.14420259964496035, 'std': 0.15926215094897586, 'sharpe': 0.6502576906002339, 'rf': 0.040641161168853454, 'n': 211}
        print(IV_ew_perf, performance_iv_ew)
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
        VW_iv_piv_leg = iv_question_c_vw_compare_legs(data)

        if show_plot:
            plot_mean_std_sr(VW_iv_piv_leg.rename(columns={'leg': 'decile', 'VW_ret':'ret'}), '5c', "VW_returns_IV_legs")
            print("In the graph, leg '-1' corresponds to bar '0'; leg '1' is bar '1'.")

        # Compare the performance of the strategy with the performance of each leg
        IV_vw_perf, performance_iv_vw = iv_question_c_vw_compare_legs_to_strat(data)
        if verbose:
            print(IV_vw_perf)
            print(performance_iv_vw)

        if True:
            mean, std, sr = get_mean_std_sharpe(EW_iv_piv_leg.rename(columns={'leg': 'decile'}))
            mean.append(performance_iv_vw['mean']), std.append(performance_iv_vw['std']), sr.append(performance_iv_vw['sharpe'])
            plot = plot_from_lists(mean, std, sr, plot_color = 'blue')
            plot.suptitle(f'Average portolio annualized mean return, standard deviation and sharpe ratio (VW_IV_legs_strat)')
            plot.savefig(f"Figures/question_5c_plot_VW_IV_legs_strat")
            plot.show()
            print("Bar 0: leg -1; Bar 1: leg 1; Bar 2: Strategy")




        # # Equally weighted strategy
        # EW_piv_, EW_piv_perf = iv_question_c_ew(data, show_plot = False) # performances of the long-short strategy

        # # Value weighted strategy
        # iv_question_c_vw(data, show_plot=True)

        # print(EW_piv_perf)


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