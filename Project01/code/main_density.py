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

n_runs = 10**7
step = 1000

# Run mcmc
for rho_star in rho_stars:
    start_time = time.time()
    print("====================")
    print(f"rho_star = {rho_star}, T_star = {T_star}, N = {N}")

    L = math.sqrt(N / rho_star)
    model = Model(T_star, L, N)
    delta = 0.3  # TODO: acceptance ratio 0.3

    U = mcmc.run_mcmc(n_runs, step, model, delta)

    end_time = time.time()
    print(f"time: {end_time - start_time}s")

    # TODO: calculate g(r)

    dir_path = f"../output/density/rho_{rho_star}/"
    ioutil.save_ndarray(dir_path + "energy.npy", U)
    ioutil.save_ndarray(dir_path + "config.npy", model.r_N)
    ioutil.save_json(dir_path + "metadata.json", 
                     rho_star=rho_star, 
                     T_star=T_star, 
                     N=N, 
                     n_runs=n_runs, 
                     step=step, 
                     delta=delta)
