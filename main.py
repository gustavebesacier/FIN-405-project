from Data_handler import Data, RF_COL
# from Utils import compute_rolling_betas, bab_equally_weighted_portfolios, bab_value_weighted_portfolios, bab_question_b
from Utils import SEP, to_YYYYMM
from BaB import run_bab_part3
from Momentum import run_mom_part4
from IV import run_iv_part5
from STRAT import run_strat_part6
from Industry_Neutral import run_in_part8
from Performance import performance

import os

def main():

    all_returns = dict()

    print(SEP*8, SEP*8, sep='\n')

    # Prepare the folder for the figures
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    # Prepare data
    Data_instance = Data()               # create instance of the class Data()
    data = Data_instance.get_data()      # store the data of Data_instance

    data.sort_values(by=['permno','date'],inplace=True)

    data = to_YYYYMM(data)

    # Check pour voir si ça résoud tous les problèmes -
    data['N'] = data.groupby(['permno'])['date'].transform('count')
    data = data[data['N']>60].copy()

    show_verbose = False
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
    print(f"{SEP} Running BaB {SEP}")
    returns_BAB = run_bab_part3(data, question_a=True, question_b=True, question_cd=True, show_plot = show, save_tables=show, verbose=show_verbose)
    all_returns['BAB'] = returns_BAB

    # # 2 MOM
    print(f"{SEP} Running Mom {SEP}")
    returns_MOM = run_mom_part4(data, question_a=True, question_b=True, save_tables=False, show_plot = show, verbose=show_verbose)
    all_returns['MOM'] = returns_MOM

    # 3 IV
    print(f"{SEP} Running IV {SEP}")
    returns_IV = run_iv_part5(data, question_a=True, question_b=True, question_c = True, show_plot = show, verbose=show_verbose)
    all_returns['IV'] = returns_IV

    # 4 STRAT
    print(SEP, SEP, SEP, SEP)
    print(f"{SEP} Running STRAT {SEP}")
    data_STRAT_RP = run_strat_part6(data, returns_BAB, returns_MOM, returns_IV, question_a=True, question_b=True, show_plot = False, verbose=True)
    ### In data_STRAT_RP, you can fetch: rBABstd, rMOMstd, rIVstd, rSTRAT (the weighted average of the strats), rSTRATstd, rFUND_RP

    # 7 Performance
    print(SEP, SEP, SEP, SEP)
    print(f"{SEP} Running Performance {SEP}")
    weights_BAB = returns_BAB['BAB_question_cd']['BAB_qc_weights']
    weights_MOM = returns_MOM['MOM_question_b']['MOM_qb_VW_weights']
    weights_IV = returns_IV['IV_question_c']['IV_question_c_VW_returns_weights']
    performance(data, weights_BAB, weights_MOM, weights_IV, data_STRAT_RP, show)


    # 8 Industry Neutral
    print(f"{SEP} Running Industry Neutral {SEP}")
    returns_STRAT_IN, data_part8_Qb  = run_in_part8(data, question_a=True, question_b=False, save_tables=False, verbose=True)


    # print(all_returns.items())
    # print(all_returns)

    # return returns_BAB, returns_MOM, returns_IV, data_STRAT_RP

    return all_returns

if __name__ == "__main__":
    main()