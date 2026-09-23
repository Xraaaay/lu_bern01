# -*- coding: utf-8 -*-
"""
Created on 2026-09-06
@author: Ruowen Xiao
"""

import math
import numpy as np
import random
from model import Model

def proposal(delta, model: Model):
    i = random.randint(0, model.N - 1)
    dx = delta * (random.random() - 0.5)
    dy = delta * (random.random() - 0.5)
    dr = np.array([dx, dy])
    return i, dr, model.cal_delta_energy(i, dr)

def acceptance(dU, model: Model):
    if dU <= 0:
        return True
    if np.isinf(dU):
        return False
    xi = random.random()
    return xi < math.exp(- dU / model.T_star)

def mcmc_step(delta, model: Model):
    i, dr, dU = proposal(delta, model)
    is_accepted = acceptance(dU, model)
    if is_accepted:
        model.move_particle(i, dr, dU)
        return True
    else:
        return False

def run_mcmc(n_runs, sample_interval_energy, sample_interval_config, model: Model, delta):
    U = []
    configs = []
    acceptance_count = 0
    for i in range(1, n_runs + 1):
        is_accepted = mcmc_step(delta, model)
        if is_accepted:
            acceptance_count += 1
        if sample_interval_energy and i % sample_interval_energy == 0:
            U.append(model.U)
        if sample_interval_config and i % sample_interval_config == 0:
            configs.append(model.r_N.copy())
    acceptance_ratio = acceptance_count / n_runs
    return U, configs, acceptance_ratio

def tune_delta(model: Model, delta_init):
    window_size = 10**5
    max_windows = 20
    delta = delta_init
    U_tune = []
    configs_tune = []

    for _ in range(max_windows):
        U, configs, acceptance_ratio = run_mcmc(window_size, None, 2 * 10**4, model, delta)
        U_tune.extend(U)
        configs_tune.extend(configs)
        print(f"acceptance ratio: {acceptance_ratio}, delta: {delta}")
        if acceptance_ratio > 0.7:
            delta *= 1.5
        elif acceptance_ratio > 0.35:
            delta *= 1.2
        elif acceptance_ratio < 0.15:
            delta *= 0.7
        elif acceptance_ratio < 0.25:
            delta *= 0.9
        else:
            break
    return delta, U_tune, configs_tune
