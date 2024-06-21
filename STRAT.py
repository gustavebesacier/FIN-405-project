import numpy as np
import pandas as pd
from Data_handler import RF_COL
from termcolor import colored

VOL_TARGET = 0.1

def get_mean_std_sharpe_STRAT(data, return_col:str):
    mean = data[return_col].mean() * 12
    std = data[return_col].std() * np.sqrt(12)
    rf = data[RF_COL].mean() * 12
    sr = (mean - rf) / std
    return mean, std, sr

def run_strat_part6(data, returns_BAB, returns_MOM, returns_IV, question_a=True, question_b=True, show_plot = False, verbose=True):
    # Get the value weighted returns for each of the strategies implemented
    dataBAB = returns_BAB[['date', 'ret']].rename(columns={'ret': 'rBAB'})
    dataMOM = returns_MOM[['date', 'ret']].rename(columns={'ret': 'rMOM'})
    dataIV = returns_IV['IV_question_c']['IV_question_c_VW_returns_data'][['date', 'ret']].rename(columns={'ret': 'rIV'})

    # Merge the datasets into one 
    dataSTRAT = pd.merge(dataBAB, dataMOM, on='date', how='inner')
    dataSTRAT = pd.merge(dataSTRAT, dataIV, on='date', how='inner')
    dataSTRAT= dataSTRAT[['date', RF_COL, 'rBAB', 'rMOM', 'rIV']]
   
    ## EQUALLY WEIGHTED 
    dataSTRAT['rSTRAT_EW'] = (dataSTRAT['rBAB'] + dataSTRAT['rMOM'] + dataSTRAT['rIV']) / 3

    # Determining the c constant for each year
    dataSTRAT['rSTRATstd'] = dataSTRAT.groupby(dataSTRAT.date.dt.year)['rSTRAT_EW'].transform('std') * np.sqrt(12)
    dataSTRAT['C'] = 0.1 / dataSTRAT['rSTRATstd']
    dataSTRAT['rFUND_EW'] = dataSTRAT['C'] * dataSTRAT['rSTRAT_EW'] + dataSTRAT[RF_COL]

    # Get the stats for the EW strategy
    retSTRAT_EW, stdSTRAT_EW, srSTRAT_EW = get_mean_std_sharpe_STRAT(dataSTRAT, 'rFUND_EW')

    print(colored("STRAT strategy based on equally weighted portfolios", attrs=['underline', 'bold'])) 
    print(" - Expected return:\t {:.2f}".format(retSTRAT_EW))
    print(" - Standard deviation:\t {:.2f}".format(stdSTRAT_EW))
    print(" - Sharpe ratio:\t {:.2f}".format(srSTRAT_EW))

    dataSTRAT.drop(columns=['rSTRAT_EW', 'rSTRATstd', 'C'])

    ### RISK PARITY 
    dataSTRAT['rBABstd'] = dataSTRAT['rBAB'].rolling(window=36, min_periods=36, closed='left').std() * np.sqrt(12)
    dataSTRAT['rMOMstd'] = dataSTRAT['rMOM'].rolling(window=36, min_periods=36, closed='left').std() * np.sqrt(12)
    dataSTRAT['rIVstd'] = dataSTRAT['rIV'].rolling(window=36, min_periods=36, closed='left').std()   * np.sqrt(12)
    dataSTRAT.dropna(inplace=True)

    dataSTRAT_RP['rSTRAT'] = dataSTRAT_RP['rBAB']/dataSTRAT_RP['rBABstd'] + dataSTRAT_RP['rMOM']/dataSTRAT_RP['rMOMstd'] + dataSTRAT_RP['rIV']/dataSTRAT_RP['rIVstd']

    dataSTRAT_RP.head()
    
    return 0