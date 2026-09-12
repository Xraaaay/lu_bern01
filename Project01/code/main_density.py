# -*- coding: utf-8 -*-
"""
Created on 2026-09-06
@author: Ruowen Xiao
"""

import ioutil
import math
import mcmc
import time
from model import Model

# Settings
rho_stars = [0.1, 0.15, 0.227, 0.291, 0.38]
T_star = 0.1
N = 100

n_equi_runs = 2 * 10**6
n_prod_runs = 2 * 10**7
sample_interval = 1000

# Run mcmc
for rho_star in rho_stars:
    start_time = time.time()
    print("====================")
    print(f"rho_star = {rho_star}, T_star = {T_star}, N = {N}")

    L = math.sqrt(N / rho_star)
    model = Model(T_star, L, N)
    delta_init = 0.3

    delta = mcmc.tune_delta(model, delta_init)
    U_equi, acceptance_ratio_equi = mcmc.run_mcmc(n_equi_runs, sample_interval, model, delta)
    print(f"acceptance raito in equilibrium: {acceptance_ratio_equi}")

    U_prod, acceptance_ratio_prod = mcmc.run_mcmc(n_prod_runs, sample_interval, model, delta)
    print(f"acceptance raito in production: {acceptance_ratio_prod}")

    end_time = time.time()
    print(f"time: {end_time - start_time}s")

    dir_path = f"../output/density/rho_{rho_star}/"
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
