from Utils import compute_rolling_betas, compute_equal_weight_data, compute_value_weighted_data, compute_ew_from_legs_data, SEP,  VERBOSE
from Grapher import plot_mean_std_sr
from Momentum import mom_add_legs

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

def iv_question_b_ew(data, verbose=VERBOSE):

    EW_iv_piv = compute_equal_weight_data(data, col_ret='ret', col_decile='decile')

    return EW_iv_piv

def iv_question_b_vw(data, verbose=VERBOSE):

    data['VW_weight'] = data.groupby(['date', 'decile'])['mcap'].transform(lambda x: x / x.sum())
    data['VW_ret_contrib'] = data['VW_weight'] * data['ret']

    VW_iv_piv = compute_value_weighted_data(data, col_ret='VW_ret_contrib', col_decile='decile').rename(columns={'VW_ret_contrib': 'ret'})

    return VW_iv_piv


def run_iv_part5(data, question_a=True, question_b=True, question_c = True, verbose=VERBOSE):
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

        # Plot the results for the 2 different weightings
        plot_mean_std_sr(EW_iv_piv, '5b', "EW_returns_IV")
        plot_mean_std_sr(VW_iv_piv, '5b', "VW_returns_IV")
        
    if question_c:

        data = mom_add_legs(data)
        data = data.dropna().copy()
        print(data)

        # For EW portfolios
        EW_iv_piv = compute_ew_from_legs_data(data, col_leg='leg', col_ret='ret')
        print("THIS IS EW IS PIV")
        print(EW_iv_piv)
        plot_mean_std_sr(EW_iv_piv.rename(columns={'leg': 'decile'}), '5c', "EW_returns_IV_legs")
        print("In the graph, leg '-1' corresponds to bar '0'; leg '1' is bar '1'.")

    return data