# -*- coding: utf-8 -*-
"""
Created on 2026-09-11
@author: Ruowen Xiao
"""

import ioutil
import math
import mcmc
import numpy as np
import time
from model import Model

# Settings
rho_star = 0.291
T_stars = [x / 100 for x in range(25, 14, -1)]
N = 100
L = math.sqrt(N / rho_star)
r_N = np.array([])
delta = 0.3

n_equi_runs = 2 * 10**6
n_prod_runs = 2 * 10**7
sample_interval = 1000

# Run mcmc
for T_star in T_stars:
    start_time = time.time()
    print("====================")
    print(f"rho_star = {rho_star}, T_star = {T_star}, N = {N}")

    model = Model(T_star, L, N, r_N)

    delta = mcmc.tune_delta(model, delta)
    U_equi, acceptance_ratio_equi = mcmc.run_mcmc(n_equi_runs, sample_interval, model, delta)
    print(f"acceptance raito in equilibrium: {acceptance_ratio_equi}")

    U_prod, acceptance_ratio_prod = mcmc.run_mcmc(n_prod_runs, sample_interval, model, delta)
    print(f"acceptance raito in production: {acceptance_ratio_prod}")

    r_N = model.r_N

    end_time = time.time()
    print(f"time: {end_time - start_time}s")

    dir_path = f"../output/temperature/T_{T_star}/"
    ioutil.save_ndarray(dir_path + "energy_equi.npy", U_equi)
    ioutil.save_ndarray(dir_path + "energy_prod.npy", U_prod)
    ioutil.save_ndarray(dir_path + "final_config.npy", model.r_N)
    ioutil.save_json(dir_path + "metadata.json", 
                     rho_star=rho_star, 
                     T_star=T_star, 
                     N=N, 
                     n_equilibrium_runs=n_equi_runs,
                     n_production_runs=n_prod_runs,
                     sample_interval=sample_interval, 
                     delta=delta)
