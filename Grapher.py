from Data_handler import RF_COL

import os

import numpy as np
import matplotlib.pyplot as plt


def get_mean_std_sharpe(data):
    """Compute the annulalized mean, standard deviation and Sharpe ratio for each decile."""
    mean = list(map(lambda x: 12*x, data.groupby('decile')['ret'].mean().values.tolist()))
    std = list(map(lambda x: np.sqrt(12)*x, data.groupby('decile')['ret'].std().values.tolist()))
    rf = np.mean(list(map(lambda x: 12*x, data.groupby('decile')[RF_COL].mean().values.tolist())))
    sr = list(map(lambda ret, vol: (ret - rf) / vol, mean, std))
    
    return mean, std, sr

def plot_from_lists(mean, std, sharpe, plot_color = 'blue'):
    deciles = list(range(len(mean)))

    a, axs = plt.subplots(1, 3, figsize=(25, 7), sharey=False)

    axs[0].bar(deciles, mean, color=plot_color)
    axs[0].set_title("Average portolio mean return")
    axs[0].set_xticks(deciles)
    axs[0].set_xlabel("Decile")
    axs[0].set_ylabel("Annualized return")

    axs[1].bar(deciles, std, color=plot_color)
    axs[1].set_title("Average portolio annualized standard deviation")
    axs[1].set_xticks(deciles)
    axs[1].set_xlabel("Decile")
    axs[1].set_ylabel("Annualized standard deviation")

    axs[2].bar(deciles, sharpe, color=plot_color)
    axs[2].set_title("Average portolio annualized sharpe ratio")
    axs[2].set_xticks(deciles)
    axs[2].set_xlabel("Decile")
    axs[2].set_ylabel("Annualized sharpe ratio")
    
    return plt

def plot_mean_std_sr(data, question, plot_name, show = True):
    """Takes a dataframe in input and returns a 3 graphs for the  annualized mean, std and sharpe ratio for some deciles."""

    if not os.path.exists("Figures"):
            os.makedirs("Figures")
    
    mean, std, sharpe = get_mean_std_sharpe(data)

    plot = plot_from_lists(mean, std, sharpe, plot_color = 'blue')

    if show:
        plot.suptitle(f'Average portolio annualized mean return, standard deviation and sharpe ratio ({plot_name})')
        plot.savefig(f"Figures/question_{question}_plot_{plot_name}")
        plot.show()