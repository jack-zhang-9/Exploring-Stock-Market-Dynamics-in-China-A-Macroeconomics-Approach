# -*- coding:utf-8 -*-
"""
作者: 张知遥
日期: 2024年04月16日
"""
from numba import njit
import numpy as np
from Model_Economy import InvSubsEcon
import matplotlib.pyplot as plt

@njit
def shooting(econ, c0, k0, T=10):
    '''
    Given the initial condition of capital k0 and an initial guess
    of consumption c0, computes the whole paths of c and k
    using the state transition law and Euler equation for T periods.
    '''
    if c0 > econ.f(k0) + (1 - econ.δ) * k0:
        print("initial consumption is not feasible")

        return None

    # initialize vectors of c and k
    c_vec = np.empty(T+1)
    k_vec = np.empty(T+2)

    c_vec[0] = c0
    k_vec[0] = k0

    for t in range(T):
        k_vec[t+1], c_vec[t+1] = econ.next(k_vec[t], c_vec[t])

    k_vec[T+1] = econ.f(k_vec[T]) + (1 - econ.δ) * k_vec[T] - c_vec[T]

    return c_vec, k_vec