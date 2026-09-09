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
    # TODO: confirm the expression to make acceptance ratio at 0.3
    dx = delta * (random.random() - 0.5)
    dy = delta * (random.random() - 0.5)
    dr = np.array([dx, dy])
    return i, dr, model.cal_delta_energy(i, dr)

def acceptance(dU, model: Model):
    if dU <= 0:
        return True
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
