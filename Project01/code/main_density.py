# -*- coding: utf-8 -*-
"""
Created on 2026-09-06
@author: Ruowen Xiao
"""

import math
import matplotlib.pyplot as plt
import mcmc
from model import Model

rho_stars = [0.1, 0.15, 0.227, 0.291, 0.38]
T_star = 0.1
N = 1000

n_runs = 10**7
step = 1000

for rho_star in rho_stars:
    L = math.sqrt(N / rho_star)
    model = Model(T_star, L, N)

    delta = 0.3
    U_mean = []
    for i in range(n_runs):
        mcmc.mcmc_step(delta, model)
        if i % step == 0:
            U_mean.append(model.U)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(0, n_runs, step), U_mean)
    ax2.scatter(*model.r_N.T)

    # TODO: calculate g(r)
    # TODO: save configs in files

plt.show()
