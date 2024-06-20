from Data_handler import Data, RF_COL
from Utils import compute_rolling_betas, bab_equally_weighted_portfolios, bab_value_weighted_portfolios, bab_question_b
from BaB import run_bab_part3

import os

def main():

    # Prepare data
    Data_instance = Data()               # create instance of the class Data()
    data = Data_instance.get_data()      # store the data of Data_instance

    data.sort_values(by=['permno','date'],inplace=True)
    
    # Add excess returns columns
    data['Rn_e'] = data['ret'] - data[RF_COL]
    data['Rm_e'] = data['vwretd'] - data[RF_COL]

    # Drop the NaN values
    print('Initial shape of the data:', data.shape)
    data = data.dropna().copy()
    print('Final shape of the data:', data.shape)

    # Get views form the data
    print('Shape data:', data.shape)
    print(data)
    print(data.columns.tolist())

    
    run_bab_part3(data)

    # Prepare the folder for the figures
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

if __name__ == "__main__":
    main()