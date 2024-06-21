from Data_handler import Data, RF_COL
# from Utils import compute_rolling_betas, bab_equally_weighted_portfolios, bab_value_weighted_portfolios, bab_question_b
from Utils import SEP
from BaB import run_bab_part3
from Momentum import run_mom_part4
from IV import run_iv_part5

import os

def main():

    all_returns = dict()

    # Prepare data
    Data_instance = Data()               # create instance of the class Data()
    data = Data_instance.get_data()      # store the data of Data_instance

    data.sort_values(by=['permno','date'],inplace=True)

    show = False
    
    # Add excess returns columns
    data['Rn_e'] = data['ret'] - data[RF_COL]
    data['Rm_e'] = data['vwretd'] - data[RF_COL]

    # Drop the NaN values
    # print('Initial shape of the data:', data.shape)
    data = data.dropna().copy()
    # print('Final shape of the data:', data.shape)

    # Get views form the data
    # print('Shape data:', data.shape)
    # print(data)

    # 1 BAB
    # print(f"{SEP} Running BaB {SEP}")
    returns_BAB = run_bab_part3(data, question_a=True, question_b=True, question_cd=True, show_plot = show, save_tables=show, verbose=show)
    all_returns['BAB'] = returns_BAB

    # 2 MOM
    print(f"{SEP} Running Mom {SEP}")
    returns_MOM = run_mom_part4(data, question_a=True, question_b=True, save_tables=False, verbose=True)
    all_returns['MOM'] = returns_MOM

    # 3 IV
    # print(f"{SEP} Running IV {SEP}")
    returns_IV = run_iv_part5(data, question_a=True, question_b=True, question_c = True, show_plot = show, verbose=show)

    all_returns['IV'] = returns_IV

    # Prepare the folder for the figures
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    return all_returns

if __name__ == "__main__":
    main()