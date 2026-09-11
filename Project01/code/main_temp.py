# -*- coding: utf-8 -*-
"""
Created on 2026-09-11
@author: Ruowen Xiao
"""

import math
import matplotlib.pyplot as plt
import mcmc
import numpy as np
from model import Model

rho_star = 0.291
T_stars = [x / 100 for x in range(25, 14, -1)]
N = 1000
L = math.sqrt(N / rho_star)
r_N = None

n_equilibrium_runs = 10**6
n_production_runs = 10**7
step = 1000

def run_mcmc(n_runs, model, delta):
    U = []
    for i in range(n_runs):
        mcmc.mcmc_step(delta, model)
        if i % step == 0:
            U.append(model.U)
    return U

for T_star in T_stars:
    model = Model(T_star, L, N, r_N)
    delta = 0.3  # TODO: acceptance ration 0.3

    U_equi = run_mcmc(n_equilibrium_runs, model, delta)
    U_prod = run_mcmc(n_production_runs, model, delta)
    fluctuation = np.mean(np.square(U_prod)) - np.square(np.mean(U_prod))
