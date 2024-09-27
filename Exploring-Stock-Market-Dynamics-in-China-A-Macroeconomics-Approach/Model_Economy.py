# -*- coding:utf-8 -*-
"""
作者: 张知遥
日期: 2024年04月16日
"""
from numba import float64
from numba.experimental import jitclass
import numpy as np


parameters2 = [
    ('γ', float64),    # Coefficient of relative risk aversion
    ('β', float64),    # Discount factor
    ('τ', float64),    # Subsidy factor
    ('δ', float64),    # Depreciation rate on capital
    ('α', float64),    # Return to capital per capita
    ('A', float64),    # Technology
    ('k_ss', float64),
    ('v_ss', float64)
]

@jitclass(parameters2)
class InvSubsEcon():

    def __init__(self, γ=2, β=0.98, τ=0, δ=0.03, α=0.5, A=1):

        self.γ, self.β = γ, β
        self.τ = τ
        self.δ, self.α, self.A = δ, α, A

        mpk = (1 / β - 1 + δ) * (1 - τ)
        k = (mpk / α) ** (1 / (α - 1))
        v = (1 - τ) * k
        self.k_ss, self.v_ss = k, v



    def u(self, c):
        '''
        Utility function
        ASIDE: If you have a utility function that is hard to solve by hand
        you can use automatic or symbolic differentiation
        See https://github.com/HIPS/autograd
        '''
        γ = self.γ

        return c ** (1 - γ) / (1 - γ) if γ!= 1 else np.log(c)

    def u_prime(self, c):
        'Derivative of utility'
        γ = self.γ

        return c ** (-γ)

    def u_prime_inv(self, c):
        'Inverse of derivative of utility'
        γ = self.γ

        return c ** (-1 / γ)

    def f(self, k):
        'Production function'
        α, A = self.α, self.A

        return A * k ** α

    def f_prime(self, k):
        'Derivative of production function'
        α, A = self.α, self.A

        return α * A * k ** (α - 1)

    def f_prime_inv(self, k):
        'Inverse of derivative of production function'
        α, A = self.α, self.A

        return (k / (A * α)) ** (1 / (α - 1))

    def next(self, k, c):
        ''''
        Given the current capital Kt and an arbitrary feasible
        consumption choice Ct, computes Kt+1 by state transition law
        and optimal Ct+1 by Euler equation.
        '''
        β, δ, τ = self.β, self.δ, self.τ
        α = self.α
        u_prime, u_prime_inv = self.u_prime, self.u_prime_inv
        f, f_prime = self.f, self.f_prime


        k_next = f(k) + (1 - δ) * k - c
        c_next = u_prime_inv(u_prime(c) / (β * (f_prime(k_next)/(1 - τ) + (1 - δ))))

        return k_next, c_next