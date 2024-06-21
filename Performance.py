import re
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from Utils import compute_rolling_betas

def load_factors(data):
    #We need to load the data from the 12 industry portfolio return and the Fama/French 5 factors
    famafrench = pd.read_csv("Data/F-F_Research_Data_5_Factors_2x3.CSV", skiprows=3)
    #Industry_weights_12 = pd.read_csv("Data/12_Industry_Portfolios.CSV", skiprows=11)
    # Load the entire CSV file
    with open("Data/12_Industry_Portfolios.CSV", 'r') as file:
        lines = file.readlines()

    # Find the start and end of the first table
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if 'Average Value Weighted Returns -- Monthly' in line:
            start_index = i + 1  # The data starts after this line
        elif start_index is not None and line.strip() == '':
            end_index = i
            break

    # Extract the lines for the first table
    table_lines = lines[start_index:end_index]

    # Create a new DataFrame from the extracted lines
    Industry_weights_12 = pd.DataFrame([x.strip().split(',') for x in table_lines[1:]], columns=table_lines[0].strip().split(','))

    # Convert appropriate columns to numeric
    Industry_weights_12 = Industry_weights_12.apply(pd.to_numeric)
    with open("Data/Siccodes12.txt", 'r') as file:
        Industries = file.read()

    # Regular expression to match each class and its indices
    pattern = r'(\d+)\s+(\w+)\s+([^\n]+)\n((?:\s+\d+-\d+\n?)+)'

    # Find all matches in the content
    matches = re.findall(pattern, Industries)

    # Initialize an empty dictionary to hold the classes and their indices
    classes = {}

    # Iterate over each match and populate the dictionary
    for match in matches:
        class_number = int(match[0])
        class_code = match[1]
        class_description = match[2].strip()
        indices_raw = match[3].strip().split('\n')

        # Extract index ranges and convert them to tuples of integers
        indices = []
        for index_range in indices_raw:
            start, end = map(int, index_range.strip().split('-'))
            indices.append((start, end))

        # Add the class to the dictionary
        classes[class_number] = {
            'code': class_code,
            'description': class_description,
            'indices': indices
        }

    # Display the classes dictionary
    for class_number, class_info in classes.items():
        print(f"Class {class_number} ({class_info['code']}): {class_info['description']}")
        for index_range in class_info['indices']:
            print(f"  Index range: {index_range[0]}-{index_range[1]}")

    def get_class_for_siccd(siccd):
        for class_number, class_info in classes.items():
            for index_range in class_info['indices']:
                if index_range[0] <= int(siccd) <= index_range[1]:
                    return class_number
        return 12

    # Apply our rule to find classes
    data['class'] = data['siccd'].apply(get_class_for_siccd)

    print(data.head())

    return famafrench, Industry_weights_12, data

def format_factors(dataSTRAT_RP,Industry_weights_12, famafrench):
    #STRAT
    RP_strategy_ret = dataSTRAT_RP[["date", "rFUND_RP"]]

    # Extract the year and month from the date column
    #RP_strategy_ret["date"] = (RP_strategy_ret["date"].dt.year * 100 + RP_strategy_ret["date"].dt.month).astype("int64")

    print(RP_strategy_ret.head())

    #Industry
    Industry_weights_12.columns.values[0] = "date"
    # The returns are in percent, scale them
    Industries_12 = Industry_weights_12.columns.drop('date')
    # Divide each column by 100
    Industry_weights_12[Industries_12] = Industry_weights_12[Industries_12] / 100

    print(Industry_weights_12.head())

    #Fama/French
    famafrench.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    # The returns are in percent, scale them
    ff_factors = famafrench.columns.drop("date")
    print(ff_factors)
    # Divide each column by 100
    famafrench[ff_factors] = famafrench[ff_factors] / 100

    print(famafrench.head())

    return RP_strategy_ret, Industry_weights_12, famafrench

def run_regression(RP_strategy_ret, Industry_weights_12, famafrench):
    # Merge
    strat_industry = pd.merge(RP_strategy_ret, Industry_weights_12, on="date", how="left")
    tmp = pd.merge(strat_industry, famafrench, on="date", how="left")

    tmp["const"] = 1
    tmp = tmp.dropna()

    regression_factors = tmp.columns.drop(["date", "rFUND_RP"])

    # Industry Exposure
    RegOLS = sm.OLS(tmp["rFUND_RP"], tmp[regression_factors]).fit()
    print("Regression coefficients and t-values for the regression on the 12 industries returns and the fama/french 5 factors")
    print(pd.concat([RegOLS.params, RegOLS.tvalues], axis=1))
    print("r-squared of the regression")
    print(RegOLS.rsquared)

    return None

def performance(data, weights_BAB, weights_MOM, weights_IV, dataSTRAT_RP,show):
    print("Question 7a\n")
    famafrench, Industry_weights_12, data = load_factors(data)
    #formatted for merging and regression
    RP_strategy_ret, Industry_weights_12, famafrench = format_factors(dataSTRAT_RP,Industry_weights_12, famafrench)
    #Run regression
    run_regression(RP_strategy_ret, Industry_weights_12, famafrench)

    print("Question 7b\n")
    # Calculate weights_BAB net weight (wL - wH)
    weights_BAB['w_BAB'] = weights_BAB["w_L"] - weights_BAB['w_H']

    # Rename the 'VW_w' columns
    weights_MOM.rename(columns={'VW_w': 'w_MOM'}, inplace=True)
    weights_IV.rename(columns={'VW_w': 'w_IV'}, inplace=True)

    # Merge weights_BAB, weights_MOM, weights_IV on permno and date
    merged_weights = weights_BAB[['permno', 'date', 'w_BAB']].merge(
        weights_MOM[['permno', 'date', 'w_MOM']], on=['permno', 'date'], how='inner'
    ).merge(
        weights_IV[['permno', 'date', 'w_IV']], on=['permno', 'date'], how='inner'
    )
    #merged_weights.dropna(inplace = True)

    print(merged_weights)

    weights_STRAT = dataSTRAT_RP[["date", 'rBABstd', 'rMOMstd', 'rIVstd']]

    # Merge with weights_STRAT on date
    final_weights = merged_weights.merge(weights_STRAT, on='date', how='inner')

    # Calculate the weight for each stock and date
    final_weights['w_stock'] = (final_weights['w_BAB'] / final_weights['rBABstd']) + \
                               (final_weights['w_MOM'] / final_weights['rMOMstd']) + \
                               (final_weights['w_IV'] / final_weights['rIVstd'])

    weights_stocks = final_weights[['permno', 'date', 'w_stock']]

    # Display the final data
    print(weights_stocks)
    data_betas = compute_rolling_betas(data, window_size=36)
    data_betas.dropna(inplace=True)
    weights_stocks_betas = weights_stocks.merge(data_betas[['permno', 'date', 'beta']], on=['permno', 'date'], how='inner')


    # Count total number of NaNs in the DataFrame
    total_nans = weights_stocks_betas.isna().sum().sum()
    print(f'Total NaNs in the DataFrame: {total_nans}')

    # Generate dummy variables for the classes
    dummies = pd.get_dummies(data_betas['class'], prefix='class').astype(int)

    # Concatenate the original DataFrame with the dummy variables
    data_with_dummies = pd.concat([data_betas, dummies], axis=1)

    #create the full data with weights industries and dummies
    data_for_exposure = data_with_dummies.merge(weights_stocks, on=['permno', 'date'], how='inner')

    colonne = data_for_exposure.columns.drop(["permno","date","ret","shrout","prc","siccd","mcap","mcap_l","tmytm","vwretd","N","Rn_e","Rm_e","class","w_stock"])

    # Estimate Factor
    Factors_tstats = data_for_exposure.groupby(['date']).apply(lambda x: sm.OLS(x['Rn_e'], x[colonne]).fit().tvalues)

    # Plot T-stats
    Factors_tstats.plot(), plt.title('T-stats Factors')
    if show:
        plt.savefig('Figures/T_stats_Factors', dpi=300)
        plt.show()

    Factors = data_for_exposure.groupby(['date']).apply(lambda x: sm.OLS(x['Rn_e'], x[colonne]).fit().params)


    Factors.plot(), plt.title('Time-Series of Factor Returns')

    if show:
        plt.savefig('Figures/TS_Factors', dpi=300)
        plt.show()

    print(pd.concat([Factors_tstats.mean(), np.abs(Factors_tstats).mean()], axis=1))

    colonne = data_for_exposure.columns.drop(
        ["permno", "ret", "shrout", "prc", "siccd", "mcap", "mcap_l", "tmytm", "vwretd", "N", "Rn_e", "Rm_e",
         "class"])

    Industries = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12']

    Exposures = data_for_exposure[colonne].copy()
    print(Exposures.columns)
    Exposures[Industries] = Exposures[Industries] * Exposures['w_stock'].to_numpy()[:, np.newaxis]
    Exposures['beta'] = Exposures['w_stock'] * Exposures['beta']
    Exposures = Exposures.groupby('date')[['beta'] + Industries].sum()

    # Plot Exposure to Tech and Finance
    plt.plot(Exposures['class_3'].rolling(36).mean())
    plt.plot(Exposures['class_6'].rolling(36).mean())
    plt.plot(Exposures['class_12'].rolling(36).mean())
    plt.legend(['Manufacturing', "Chems", "Others"])
    if show:
        plt.savefig('Figures/Industries_3', dpi=300)
        plt.show()

    print("Question 7c\n")

    # Compute Hedge Portfolio Return
    Hedge_Return = Factors.mul(Exposures, axis=1).sum(axis=1)
    print(Hedge_Return)
    print("exposures",Exposures)
    print("factors",Factors)

    # Compute the industry-hedged STRAT return by subtracting the hedge return from the original STRAT return
    STRAT_returns = data_for_exposure.groupby('date')['beta'].mean()  # Assuming Rn_e is the STRAT return
    STRAT_hedged_return = STRAT_returns - Hedge_Return

    # Calculate average return, standard deviation, and Sharpe ratio of the industry-hedged STRAT return
    average_return = STRAT_hedged_return.mean() * 12  # Annualize the mean return
    std_return = STRAT_hedged_return.std() * np.sqrt(12)  # Annualize the standard deviation
    sharpe_ratio = average_return / std_return
    print(STRAT_hedged_return)

    # Print the results
    print('Industry-Hedged STRAT Return: ', average_return)
    print('Industry-Hedged STRAT Std: ', std_return)
    print('Industry-Hedged STRAT Sharpe: ', sharpe_ratio)

    # Optionally, you can plot the hedged return time series
    if show:
        STRAT_hedged_return.plot()
        plt.title('Industry-Hedged STRAT Return')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.savefig('Figures/Industry_Hedged_STRAT_Return', dpi=300)
        plt.show()
