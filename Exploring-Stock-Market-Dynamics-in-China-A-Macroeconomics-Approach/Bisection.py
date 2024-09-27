# -*- coding:utf-8 -*-
"""
作者: 张知遥
日期: 2024年04月16日
"""
from numba import njit
from Shooting_Algorithm import shooting
import matplotlib.pyplot as plt
import numpy as np
from Model_Economy import InvSubsEcon

@njit
def bisection(econ, c0, k0, T=10, tol=1e-4, max_iter=500, k_ter=0, verbose=True):

    # initial boundaries for guess c0
    c0_upper = econ.f(k0) + (1 - econ.δ) * k0
    c0_lower = 0

    i = 0
    while True:
        c_vec, k_vec = shooting(econ, c0, k0, T)
        error = k_vec[-1] - k_ter

        # check if the terminal condition is satisfied
        if np.abs(error) < tol:
            if verbose:
                print('Converged successfully on iteration ', i+1)
            return c_vec, k_vec

        i += 1
        if i == max_iter:
            if verbose:
                print('Convergence failed.')
            return c_vec, k_vec

        # if iteration continues, updates boundaries and guess of c0
        if error > 0:
            c0_lower = c0
        else:
            c0_upper = c0

        c0 = (c0_lower + c0_upper) / 2