import pandas as pd
import numpy as np
from BaB import run_bab_part3
from Momentum import run_mom_part4
from IV import run_iv_part5
from Performance import load_factors
from Utils import SEP

### CALLED NOWHERE FOR NOW, WILL START AGAIN LATER

def separation_per_industry(data):
    data_separated = {}
    famafrench, Industry_weights_12, data = load_factors(data)
    for i in range(1, 12):
       data_separated[i-1] = data[data['class'] == i]

    return data_separated

def compute_centered_t_stat(x,std,n):
    return x/(std/np.sqrt(n))


def run_in_part8(data, question_a=True, question_b=False, question_c= True, save_tables=False, verbose=True):
    if question_a:
        data_separated = separation_per_industry(data)
        returns_BAB_IN = {}
        returns_MoM_IN = {}
        returns_IV_IN = {}
        returns_STRAT_IN = {}

        show = False
        show_verbose = True


        for i in range(11):
            print(SEP, f"Industry {i}", SEP)
            returns_BAB_IN[i] = run_bab_part3(data_separated[i], question_a=True, question_b=True, question_cd=True, show_plot = show, save_tables=show, verbose=show_verbose)
            returns_MoM_IN[i] = run_mom_part4(data_separated[i], question_a=True, question_b=True, save_tables=False, show_plot = show, verbose=show_verbose)#run_mom_part4(data_separated[i], question_a=True, question_b=True, save_tables=True, verbose=True)
            returns_IV_IN[i] = run_iv_part5(data_separated[i], question_a=True, question_b=True, question_c = True, show_plot = show, verbose=show_verbose)#run_iv_part5(data_separated[i], question_a=True, question_b=True, verbose=True)
        

    return 0