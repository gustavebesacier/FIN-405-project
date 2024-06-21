import pandas as pd
import numpy as np
from BaB import run_bab_part3
from Momentum import run_mom_part4
from IV import run_iv_part5
from Performance import load_factors
from Utils import SEP, RF_COL
from STRAT import run_strat_part6, get_mean_std_sharpe_STRAT

def separation_per_industry(data):
    data_separated = {}
    famafrench, Industry_weights_12, data = load_factors(data)
    for i in range(1, 12):
       data_separated[i-1] = data[data['class'] == i]

    return data_separated

def compute_centered_t_stat(x,std,n):
    return x/(std/np.sqrt(n))

def format_factors(Industry_weights_12, famafrench):
 
    #Industry
    Industry_weights_12.columns.values[0] = "date"
    # The returns are in percent, scale them
    Industries_12 = Industry_weights_12.columns.drop('date')
    # Divide each column by 100
    Industry_weights_12[Industries_12] = Industry_weights_12[Industries_12] / 100


    #Fama/French
    famafrench.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    # The returns are in percent, scale them
    ff_factors = famafrench.columns.drop("date")
    print(ff_factors)
    # Divide each column by 100
    famafrench[ff_factors] = famafrench[ff_factors] / 100


    return Industry_weights_12, famafrench

def run_regression(RP_strategy_ret, Industry_weights_12, famafrench):
    # Merge
    strat_industry = pd.merge(RP_strategy_ret, Industry_weights_12, on="date", how="left")
    tmp = pd.merge(strat_industry, famafrench, on="date", how="left")

    tmp["const"] = 1
    tmp = tmp.dropna()

    regression_factors = tmp.columns.drop(["date", "INstrat"])

    # Industry Exposure
    import statsmodels.api as sm
    RegOLS = sm.OLS(tmp["INstrat"], tmp[regression_factors]).fit()
    print("Regression coefficients and t-values for the regression on the 12 industries returns and the fama/french 5 factors")
    print(pd.concat([RegOLS.params, RegOLS.tvalues], axis=1))
    print("r-squared of the regression")
    print(RegOLS.rsquared)

    return None


def run_in_part8(data, rf_util, question_a=True, question_b=False, question_c= True, save_tables=False, verbose=True):
    if question_a:
        data_separated = separation_per_industry(data)
        returns_BAB_IN_temp = {}
        returns_MoM_IN_temp = {}
        returns_IV_IN_temp = {}
        returns_STRAT_IN = {}
        strat_IN_temp = {}

        show = False
        show_verbose = True


        for i in range(11):
            print(SEP, f"Industry {i}", SEP)
            returns_BAB_IN_temp[i] = run_bab_part3(data_separated[i], question_a=True, question_b=True, question_cd=True, show_plot = False, save_tables=show, verbose=show_verbose)
            returns_MoM_IN_temp[i] = run_mom_part4(data_separated[i], question_a=True, question_b=True, save_tables=False, show_plot = False, verbose=show_verbose)#run_mom_part4(data_separated[i], question_a=True, question_b=True, save_tables=True, verbose=True)
            returns_IV_IN_temp[i] = run_iv_part5(data_separated[i], question_a=True, question_b=True, question_c = True, show_plot = False, verbose=show_verbose)#run_iv_part5(data_separated[i], question_a=True, question_b=True, verbose=True)

  
        ### RUN STRAT FOR EACH INDUSTRY
        for i in range(11):
            strat_IN_temp[i] = run_strat_part6(data_separated[i], returns_BAB_IN_temp[i], returns_MoM_IN_temp[i], returns_IV_IN_temp[i], question_a=True, question_b=True, show_plot = False, verbose=True)
            returns_STRAT_IN[i] = strat_IN_temp[i]['rFUND_RP']

        TSTAT = []
        ### GET THE T STATS
        for i in range(11):
            ret = returns_STRAT_IN[i].mean()
            std = returns_STRAT_IN[i].std()
            n = len(returns_STRAT_IN[i])
            TSTAT.append(ret/ (std/np.sqrt(n)))

        print("THESE ARE THE T STATS YOU NEED")
        print(TSTAT)
        date_util = strat_IN_temp[1]['date'].astype(int)
        date_util = strat_IN_temp[1]['date'].astype(int)
    
    if question_b:
        ### EQUALLY WEIGTHED PORTFOLIO OF THE RETURNS
        data_part8_Qb = pd.DataFrame(returns_STRAT_IN)
        data_part8_Qb['INstrat'] = data_part8_Qb.mean(axis=1)
        data_part8_Qb['date'] = date_util

        data_part8_Qb[RF_COL] = rf_util
        mean_IN, std_IN, sr_IN = get_mean_std_sharpe_STRAT(data_part8_Qb, return_col='INstrat')
        
        print("\n----------------------- Industry neutral strategy")
        print(" - Mean return: {:.4f}".format(mean_IN))
        print(" - Standard deviation: {:.4f}".format(std_IN))
        print(" - Sharpe ratio: {:.4f}".format(sr_IN))

        # t_stat_IN = compute_centered_t_stat(mean_IN, std_IN, len(data_part8_Qb))
    
    if question_c:
        famafrench, Industry_weights_12, data___ = load_factors(data)
        data_to_regress = data_part8_Qb[['date', 'INstrat']]
        data_to_regress2 = data_to_regress.copy()
        Industry_weights_12, famafrench = format_factors(Industry_weights_12, famafrench)
        run_regression(data_to_regress, Industry_weights_12, famafrench)

    return returns_STRAT_IN, data_part8_Qb