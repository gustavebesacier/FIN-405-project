import numpy as np
import pandas as pd
from Data_handler import RF_COL
from scipy.optimize import minimize
from Utils import to_YYYYMM

VOL_TARGET = 0.1

def get_mean_std_sharpe_STRAT(data, return_col:str):
    mean = data[return_col].mean() * 12
    std = data[return_col].std() * np.sqrt(12)
    rf = data[RF_COL].mean() * 12
    sr = (mean - rf) / std
    return mean, std, sr

# Calculate mean-variance optimal weights
def mean_variance_weights(mean_returns, cov_matrix, RF_COL):
    num_assets = len(mean_returns)
    
    # Calculate portfolio volatility given weights
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Minimize the negative Sharpe ratio = maximize positive Sharpe ratio
    def negative_sharpe_ratio(weights):
        port_return = np.dot(weights, mean_returns)
        port_volatility = portfolio_volatility(weights)
        return -(port_return - RF_COL) / port_volatility
    
    # Weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Common initial guess for weights (equal weights)
    init_guess = num_assets * [1. / num_assets]
    
    result = minimize(negative_sharpe_ratio, init_guess, constraints=constraints)
    
   
    return result.x


def run_strat_part6(data, returns_BAB, returns_MOM, returns_IV, question_a=True, question_b=True, show_plot = False, verbose=True):
    # Get the value weighted returns for each of the strategies implemented
    dataBAB = returns_BAB['BAB_question_cd']['BAB_qc_strategy_data'][['date', 'rBAB']]#.rename(columns={'ret': 'rBAB'})
    print(dataBAB)
    dataMOM = returns_MOM['MOM_question_b']['MOM_qb_VW_returns'][['date', 'VW_ret', RF_COL]].rename(columns={'VW_ret': 'rMOM'})
    print(dataMOM)
    dataIV = returns_IV['IV_question_c']['IV_question_c_VW_returns_data'][['date', 'VW_ret']].rename(columns={'VW_ret': 'rIV'})
    print(dataIV)

    # Merge the datasets into one 
    dataSTRAT = pd.merge(dataBAB, dataMOM, on='date', how='inner')
    dataSTRAT = pd.merge(dataSTRAT, dataIV, on='date', how='inner')
    print(dataSTRAT)
    print(dataSTRAT.columns)
    dataSTRAT = dataSTRAT[['date', RF_COL, 'rBAB', 'rMOM', 'rIV']]
    print(dataSTRAT)
   

    ## EQUALLY WEIGHTED 
    dataSTRAT_EW = dataSTRAT.copy()
    dataSTRAT_EW['rSTRAT_EW'] = (dataSTRAT_EW['rBAB'] + dataSTRAT_EW['rMOM'] + dataSTRAT_EW['rIV']) / 3

    # Determining the c constant for each year
    dataSTRAT_EW["date"] = pd.to_datetime(dataSTRAT_EW["date"].astype(int), format='%Y%m')
    print("test1",dataSTRAT_EW.head())
    dataSTRAT_EW['rSTRATstd'] = dataSTRAT_EW.groupby(dataSTRAT_EW.date.dt.year)['rSTRAT_EW'].transform('std') * np.sqrt(12)
    print("test2",dataSTRAT_EW.head())
    dataSTRAT_EW = to_YYYYMM(dataSTRAT_EW)

    dataSTRAT_EW['C'] = VOL_TARGET / dataSTRAT_EW['rSTRATstd']
    dataSTRAT_EW['rFUND_EW'] = dataSTRAT_EW['C'] * dataSTRAT_EW['rSTRAT_EW'] + dataSTRAT_EW[RF_COL]

    # Get the stats for the EW strategy
    retSTRAT_EW, stdSTRAT_EW, srSTRAT_EW = get_mean_std_sharpe_STRAT(dataSTRAT_EW, 'rFUND_EW')

    print("STRAT strategy based on equally weighted portfolios") 
    print(" - Expected return:\t {:.2f}".format(retSTRAT_EW))
    print(" - Standard deviation:\t {:.2f}".format(stdSTRAT_EW))
    print(" - Sharpe ratio:\t {:.2f}".format(srSTRAT_EW))

    dataSTRAT_EW.drop(columns=['rSTRAT_EW', 'rSTRATstd', 'C'], axis=1)



    ### RISK PARITY
    dataSTRAT_RP = dataSTRAT.copy()

    dataSTRAT_RP['rBABstd'] = dataSTRAT_RP['rBAB'].rolling(window=36, min_periods=36, closed='left').std() * np.sqrt(12)
    dataSTRAT_RP['rMOMstd'] = dataSTRAT_RP['rMOM'].rolling(window=36, min_periods=36, closed='left').std() * np.sqrt(12)
    dataSTRAT_RP['rIVstd'] = dataSTRAT_RP['rIV'].rolling(window=36, min_periods=36, closed='left').std()   * np.sqrt(12)
    dataSTRAT_RP.dropna(inplace=True)

    dataSTRAT_RP['rSTRAT'] = dataSTRAT_RP['rBAB']/dataSTRAT_RP['rBABstd'] + dataSTRAT_RP['rMOM']/dataSTRAT_RP['rMOMstd'] + dataSTRAT_RP['rIV']/dataSTRAT_RP['rIVstd']

    # Determining the c constant for each year
    dataSTRAT_RP["date"] = pd.to_datetime(dataSTRAT_RP["date"].astype(int), format='%Y%m')
    dataSTRAT_RP['rSTRATstd'] = dataSTRAT_RP.groupby(dataSTRAT_RP.date.dt.year)['rSTRAT'].transform('std') * np.sqrt(12)
    dataSTRAT_RP = to_YYYYMM(dataSTRAT_RP)
    dataSTRAT_RP['C'] = VOL_TARGET / dataSTRAT_RP['rSTRATstd']
    dataSTRAT_RP['rFUND_RP'] = dataSTRAT_RP['C'] * dataSTRAT_RP['rSTRAT'] + dataSTRAT_RP[RF_COL]

    retSTRAT_RP, stdSTRAT_RP, srSTRAT_RP = get_mean_std_sharpe_STRAT(dataSTRAT_RP, 'rFUND_RP')

    print("STRAT strategy based on equally risk parity portfolios") 
    print(" - Expected return:\t {:.2f}".format(retSTRAT_RP))
    print(" - Standard deviation:\t {:.2f}".format(stdSTRAT_RP))
    print(" - Sharpe ratio:\t {:.2f}".format(srSTRAT_RP))


    ### MEAN VARIANCE OPTIMAL
    dataSTRAT_MV = dataSTRAT.copy()
    mean_MV = dataSTRAT_MV[['rBAB', 'rIV', 'rMOM']].rolling(window=36, min_periods=36, closed='left').mean() 
    covariance_MV = dataSTRAT_MV[['rBAB', 'rIV', 'rMOM']].rolling(window=36, min_periods=36, closed='left').cov() 
    dataSTRAT_MV.dropna(inplace=True)

    optimal_weights_list = []
    portfolio_returns = []

    # Calculate optimal weights and portfolio returns for each rolling window
    for i in range(36, len(dataSTRAT_MV)):
        rolling_mean = mean_MV.iloc[i].values
        rolling_cov = covariance_MV.iloc[i*3:(i+1)*3].values.reshape(3, 3)
        
        optimal_weights = mean_variance_weights(rolling_mean, rolling_cov, dataSTRAT_MV[RF_COL].iloc[i])
        optimal_weights_list.append(optimal_weights)
        
        # Calculate portfolio return for the current period
        portfolio_return = np.dot(optimal_weights, dataSTRAT_MV[['rBAB', 'rIV', 'rMOM']].iloc[i].values)
        portfolio_returns.append(portfolio_return)

    # Store the results 
    #Ca on peut changer mais oke 
    dataSTRAT_MV = dataSTRAT_MV.iloc[36:].copy()
    dataSTRAT_MV['optimal_weights'] = optimal_weights_list
    dataSTRAT_MV['rSTRAT_MV'] = portfolio_returns

    # Determine the c constant for each year
    dataSTRAT_MV["date"] = pd.to_datetime(dataSTRAT_MV["date"].astype(int), format='%Y%m')
    dataSTRAT_MV['rSTRATstd'] = dataSTRAT_MV.groupby(dataSTRAT_MV.date.dt.year)['rSTRAT_MV'].transform('std') * np.sqrt(12)
    dataSTRAT_MV = to_YYYYMM(dataSTRAT_MV)
    dataSTRAT_MV['C'] = 0.1 / dataSTRAT_MV['rSTRATstd']
    dataSTRAT_MV['rFUND_MV'] = dataSTRAT_MV['C'] * dataSTRAT_MV['rSTRAT_MV'] + dataSTRAT_MV[RF_COL]

    retSTRAT_MV, stdSTRAT_MV, srSTRAT_MV = get_mean_std_sharpe_STRAT(dataSTRAT_MV, 'rFUND_MV')
    print("STRAT strategy based on mean-variance optimal combination")
    print(" - Expected return:\t {:.2f}".format(retSTRAT_MV))
    print(" - Standard deviation:\t {:.2f}".format(stdSTRAT_MV))
    print(" - Sharpe ratio:\t {:.2f}".format(srSTRAT_MV))
    return dataSTRAT_RP