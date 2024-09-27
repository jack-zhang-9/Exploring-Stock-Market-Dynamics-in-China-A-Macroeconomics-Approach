# -*- coding:utf-8 -*-
"""
作者: 张知遥
日期: 2024年05月05日
"""
from Model_Economy import InvSubsEcon
from Bisection import bisection
from Shooting_Algorithm import shooting
import matplotlib.pyplot as plt
import numpy as np

def plot_paths(econ_list, c_0, k0_list, T, inv_plot=False, prof_plot=False, r_plot=False, save=False, return_save=False):

    # Create a figure for each set of variables
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 4))
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4))
    if inv_plot:
        fig3, axs3 = plt.subplots(1, 2, figsize=(10, 4))
    if prof_plot:
        fig4, axs4 = plt.subplots(1, 2, figsize=(10, 4))
    if r_plot:
        fig5, axs5 = plt.subplots(1, 1, figsize=(5, 4))

    ylabels = ['$c_t$', '$k_t$']
    titles = ['Consumption', 'Capital']
    titles2 = ['Valuation', 'PB Ratio']
    titles3 = ['Investment', 'Investment Ratio']
    titles4 = ['Dividend', 'Dividend Ratio']

    line_styles = ['-', '--', '-.', ':']  # Different line styles for distinguishing between models

    for i, (econ, k_0) in enumerate(zip(econ_list, k0_list)):

        c_path, k_path = bisection(econ, c_0, k_0, T, k_ter=econ.k_ss, verbose=True)

        # Valuation
        v_path = (1 - econ.τ) * k_path[1:T+2]
        r_path = (econ.f_prime(k_path) / (1 - econ.τ) + 1 - econ.δ)
        pb_path = v_path / k_path[1:]

        # Investment
        x_path = k_path[1:T + 2] - (1 - econ.δ) * k_path[:T+1]
        x_ratio = x_path / k_path[:T+1]

        # Profit
        p_path = econ.α * econ.f(k_path[:T+1]) - (1-econ.τ) * x_path
        p_ratio = p_path / k_path[:T+1]

        paths_sum = [c_path, k_path]
        value_sum = [v_path, pb_path]
        inv_sum = [x_path, x_ratio]
        prof_sum = [p_path, p_ratio]

        # Plot Consumption and Capital paths
        for j, path in enumerate(paths_sum):
            axs1[j].plot(path, linestyle=line_styles[i], label=f"τ = {econ.τ}")
            axs1[j].set(xlabel='t', ylabel=ylabels[j], title=titles[j])
            axs1[j].legend()

        # Plot Valuation and PB Ratio
        for j, path in enumerate(value_sum):
            axs2[j].plot(path, linestyle=line_styles[i], label=f"τ = {econ.τ}")
            axs2[j].set(xlabel='t', ylabel=f"{titles2[j]}", title=titles2[j])
            axs2[j].legend()

        # Plot Investment Ratios
        if inv_plot:
            for j, path in enumerate(inv_sum):
                axs3[j].plot(path, linestyle=line_styles[i], label=f"τ = {econ.τ}")
                axs3[j].set(xlabel='t', ylabel=f"{titles3[j]}", title=titles3[j])
                axs3[j].legend()

        # Plot Profit Ratios
        if prof_plot:
            for j, path in enumerate(prof_sum):
                axs4[j].plot(path, linestyle=line_styles[i], label=f"τ = {econ.τ}")
                axs4[j].set(xlabel='t', ylabel=f"{titles4[j]}", title=titles4[j])
                axs4[j].legend()

        # Plot Return
        if r_plot:
            axs5.plot(r_path, linestyle=line_styles[i], label=f"τ = {econ.τ}")
            axs5.set(xlabel='t', ylabel='Return', title='Stock Return')
            axs5.legend()

        if save:
            fig1.savefig('Figure/Consumption and Capital.jpg')
            fig2.savefig('Figure/Valuation and PB Ratio.jpg')
            fig3.savefig('Figure/Investment.jpg')
            fig4.savefig('Figure/Dividend.jpg')

        if return_save:
            fig5.savefig('Figure/Return path.jpg')

    plt.show()


if __name__ == "__main__":
    econ1 = InvSubsEcon()
    econ2 = InvSubsEcon(τ=0.2)
    print(econ1.k_ss)
    print(econ2.k_ss)
    plot_paths([econ1, econ2], 1, [0.5 * econ1.k_ss, 0.5 * econ1.k_ss], 300, inv_plot=True, prof_plot=True, save=True)
    plot_paths([econ1, econ2], 1, [0.5 * econ1.k_ss, 0.5 * econ1.k_ss], 300, r_plot=True)
