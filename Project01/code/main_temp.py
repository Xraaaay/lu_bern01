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
T_star_init = 0.25
N = 100
L = math.sqrt(N / rho_star)
delta = 0.3

n_equi_runs = 2 * 10**6
n_prod_runs = 2 * 10**7
sample_interval_energy = 1000
sample_interval_config = 10**5

model = Model(T_star_init, L, N)

# Run mcmc
for T_star in T_stars:
    start_time = time.time()
    print("====================")
    print(f"rho_star = {rho_star}, T_star = {T_star}, N = {N}")

    model.T_star = T_star

    delta, configs_tune = mcmc.tune_delta(model, delta)
    U_equi, configs_equi, _ = mcmc.run_mcmc(n_equi_runs, sample_interval_energy, 
                                            sample_interval_config, model, delta)
    U_prod, configs_prod, _ = mcmc.run_mcmc(n_prod_runs, sample_interval_energy, 
                                            sample_interval_config, model, delta)
    configs = configs_tune + configs_equi + configs_prod

    end_time = time.time()
    print(f"time: {end_time - start_time}s")

    dir_path = f"../output/temperature/T_{T_star}/"
    ioutil.save_ndarray(dir_path + "energy_equi.npy", np.asarray(U_equi))
    ioutil.save_ndarray(dir_path + "energy_prod.npy", np.asarray(U_prod))
    ioutil.save_ndarray(dir_path + "configs.npy", configs)
    ioutil.save_ndarray(dir_path + "final_config.npy", model.r_N)
    ioutil.save_json(dir_path + "metadata.json", 
                     rho_star=rho_star, 
                     T_star=T_star, 
                     N=N, 
                     n_equilibrium_runs=n_equi_runs,
                     n_production_runs=n_prod_runs,
                     sample_interval_energy=sample_interval_energy, 
                     delta=delta)
