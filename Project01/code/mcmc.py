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
    # TODO: add occasional attempts
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

def run_mcmc(n_runs, sample_interval, model: Model, delta):
    U = []
    acceptance_count = 0
    for i in range(1, n_runs + 1):
        is_accepted = mcmc_step(delta, model)
        if is_accepted:
            acceptance_count += 1
        if i % sample_interval == 0:
            U.append(model.U)
    acceptance_ratio = acceptance_count / n_runs
    return U, acceptance_ratio

def tune_delta(model: Model, delta_init):
    window_size = 10**5
    max_windows = 20
    delta = delta_init
    for _ in range(max_windows):
        _, acceptance_ratio = run_mcmc(window_size, window_size + 1, model, delta)
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
    return delta
