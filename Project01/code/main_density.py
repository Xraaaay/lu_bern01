# -*- coding: utf-8 -*-
"""
Created on 2026-09-06
@author: Ruowen Xiao
"""

import ioutil
import math
import mcmc
import numpy as np
import time
from model import Model

# Settings
rho_stars = [0.1, 0.15, 0.227, 0.291, 0.38]
T_star = 0.1
N = 100

n_equi_runs = 2 * 10**6
n_prod_runs = 2 * 10**7
sample_interval_energy = 1000
sample_interval_config = 10**5

# Run mcmc
for rho_star in rho_stars:
    start_time = time.time()
    print("====================")
    print(f"rho_star = {rho_star}, T_star = {T_star}, N = {N}")

    L = math.sqrt(N / rho_star)
    model = Model(T_star, L, N)
    delta_init = 0.3
    U_init = [model.U]
    config_init = [model.r_N.copy()]

    delta, U_tune, configs_tune = mcmc.tune_delta(model, delta_init)
    U_equi, configs_equi, _ = mcmc.run_mcmc(n_equi_runs, sample_interval_energy, 
                                            sample_interval_config, model, delta)
    U_prod, configs_prod, _ = mcmc.run_mcmc(n_prod_runs, sample_interval_energy, 
                                            sample_interval_config, model, delta)
    U_total = U_init + U_tune + U_equi + U_prod
    configs = config_init + configs_tune + configs_equi + configs_prod

    end_time = time.time()
    print(f"time: {end_time - start_time}s")

    dir_path = f"../output/density/rho_{rho_star}/"
    ioutil.save_ndarray(dir_path + "energy.npy", np.asarray(U_total))
    ioutil.save_ndarray(dir_path + "configs.npy", configs)
    ioutil.save_ndarray(dir_path + "final_config.npy", model.r_N)
    ioutil.save_json(dir_path + "metadata.json", 
                     rho_star=rho_star, 
                     T_star=T_star, 
                     N=N, 
                     n_equilibrium_runs=n_equi_runs,
                     n_production_runs=n_prod_runs,
                     sample_interval_energy=sample_interval_energy, 
                     sample_interval_config=sample_interval_config, 
                     delta=delta)
