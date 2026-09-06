# -*- coding: utf-8 -*-
"""
Created on 2026-09-06
@author: Ruowen Xiao
"""

import numpy as np

class Model:
    def __init__(self, sigma0, temp, area, n) -> None:
        self.sigma0 = sigma0
        self.temp = temp
        self.area = area
        self.n = n
        self.state = self.init_state()

    def init_state(self):
        pass

    def interaction_energy(self, r_i, r_j, epsilon):
        sigma0 = self.sigma0
        sigma1 = 2.5 * sigma0
        r = np.linalg.norm(r_i - r_j)
        if r <= sigma0:
            potential = np.inf
        elif sigma0 < r < sigma1:
            potential = epsilon
        else:
            potential = 0
        return potential

    def total_energy(self):
        pass

    def select_particle(self):
        return np.random.randint(1, self.n + 1)

    def move_particle(self):
        pass
