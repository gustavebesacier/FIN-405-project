import pandas as pd
from BaB import run_bab_part3
from Momentum import run_mom_part4
from IV import run_iv_part5

### CALLED NOWHERE FOR NOW, WILL START AGAIN LATER

NUMBER_TO_INDUSTRY = {
    '1': 'Agriculture',
    '2': 'Mining',
    '3': 'Construction', 
    '4': 'Manufacturing',
    '5': 'Transportation',
    '6': 'Utilities',
    '7': 'Wholesale',
    '8': 'Retail',
    '9': 'Finance',
    '10': 'Services',
    '11': 'Public',
    '12': 'Missing'
    }

### This snippet of code was taken from the internet and modified. Source: https://www.tidy-finance.org/python/wrds-crsp-and-compustat.html
def industry_standardizing(siccd):
    if 1 <= siccd <= 999:
        return 1
    elif 1000 <= siccd <= 1499:
        return 2
    elif 1500 <= siccd <= 1799:
        return 3
    elif 2000 <= siccd <= 3999:
        return 4
    elif 4000 <= siccd <= 4899:
        return 5
    elif 4900 <= siccd <= 4999:
        return 6
    elif 5000 <= siccd <= 5199:
        return 7
    elif 5200 <= siccd <= 5999:
        return 8
    elif 6000 <= siccd <= 6799:
        return 9
    elif 7000 <= siccd <= 8999:
        return 10
    elif 9000 <= siccd <= 9999:
        return 11
    else:
        return 12


def separation_per_industry(data):
    data_per_industry = {}
    data['industry_number'] = data['siccd'].apply(industry_standardizing)

    for i in range(1, 12):
        data_per_industry[i] = data[data['industry_number'] == i]

    return data_per_industry


def run_industry_neutral(data):
    data_separated = separation_per_industry(data)
    returns_BAB_IN = {}
    returns_MoM_IN = {}
    returns_IV_IN = {}
    returns_STRAT_IN = {}

    for i in range(1, 12):
        returns_BAB_IN[i] = run_bab_part3(data_separated[i], question_a=True, question_b=True, question_cd=True, save_tables=True, verbose=True)
        returns_MoM_IN[i] = run_mom_part4(data_separated[i], question_a=True, question_b=True, save_tables=True, verbose=True)
        returns_IV_IN[i] = run_iv_part5(data_separated[i], question_a=True, question_b=True, verbose=True)
    

    return 0