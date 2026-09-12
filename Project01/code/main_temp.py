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

n_equi_runs = 2 * 10**6
n_prod_runs = 2 * 10**7
step = 1000

# Run mcmc
for T_star in T_stars:
    start_time = time.time()
    print("====================")
    print(f"rho_star = {rho_star}, T_star = {T_star}, N = {N}")

    model = Model(T_star, L, N, r_N)
    delta = 0.3  # TODO: acceptance ratio 0.3

    U_equi = mcmc.run_mcmc(n_equi_runs, step, model, delta, is_equi=True)
    U_prod = mcmc.run_mcmc(n_prod_runs, step, model, delta)

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
                     step=step, 
                     delta=delta)
