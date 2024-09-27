# -*- coding:utf-8 -*-
"""
作者: 张知遥
日期: 2024年04月19日
"""
import numpy as np
import matplotlib.pyplot as plt

# Given parameters
β = 0.98
α = 0.5
δ = 0.03

τ_values = np.linspace(0, 0.5, 100)

# Initialize lists to store k, v, and vk_ratio values
k_values = []
v_values = []
vk_ratio_values = []

for τ in τ_values:
    mpk = (1 / β - 1 + δ) * (1-τ)
    k = (mpk / α) ** (1 / (α - 1))
    d = (mpk - (1-τ)*δ) * k
    v = (β / (1 - β)) * d
    vk_ratio = v / k

    k_values.append(k)
    v_values.append(v)
    vk_ratio_values.append(vk_ratio)

# Create subplots
fig1, ax1 = plt.subplots(1, 1, figsize=(5, 4))
fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))

# Plot k and v values in the first subplot
ax1.plot(τ_values[:len(k_values)], k_values, label='k', color='b')
ax1.plot(τ_values[:len(v_values)], v_values, label='v', color='r',linestyle='--')
ax1.set_xlabel('τ')
ax1.set_ylabel('Value')
ax1.set_title('Comparison of k and v for different values of τ')
ax1.legend()
ax1.grid(True)
fig1.savefig('Figure/Comparison of k and v.jpg')

# Plot vk_ratio values in the second subplot
ax2.plot(τ_values[:len(vk_ratio_values)], vk_ratio_values, label='vk_ratio', color='g')
ax2.set_xlabel('τ')
ax2.set_ylabel('vk_ratio')
ax2.set_title('Comparison of vk_ratio for different values of τ')
ax2.legend()
ax2.grid(True)

# Adjust layout
plt.tight_layout()

# Show plots
plt.show()

