# -*- coding: utf-8 -*-
"""
Created on 2026-09-06
@author: Ruowen Xiao
"""

import math
import matplotlib.pyplot as plt
import mcmc
from model import Model

rho_star = 0.291
N = 100
L = math.sqrt(N / rho_star)
model = Model(T_star=0.1, L=L, N=N)

n_runs = 10**7
step = 1000
delta = 0.3
U_mean = []
for i in range(n_runs):
    mcmc.mcmc_step(delta, model)
    if i % step == 0:
        U_mean.append(model.U)

fig1, ax1 = plt.subplots()
ax1.plot(range(0, n_runs, step), U_mean)

fig2, ax2 = plt.subplots()
ax2.scatter(*model.r_N.T)
plt.show()
