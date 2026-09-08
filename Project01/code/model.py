# -*- coding: utf-8 -*-
"""
Created on 2026-09-06
@author: Ruowen Xiao
"""

import math
import numpy as np

class Model:
    def __init__(self, T_star, L, N) -> None:
        self.T_star = T_star  # T_star = k * T / epsilon
        self.L = L  # length of box
        self.N = N  # number of particles
        self.sigma0 = 1  # radius of hard core
        self.sigma1 = 2.5 * self.sigma0  # radius of soft core
        self.r_N = self._cold_start()
        self.U = self.cal_total_energy()

    def _cold_start(self):
        rows = math.floor(math.sqrt(self.N))
        columns = self.N // rows
        remainder = self.N % rows

        dy = self.L / rows
        dx1 = self.L / columns
        dx2 = self.L / (columns + 1)

        coordinates = []
        for i in range(rows):
            if remainder > 0:
                c = columns + 1
                dx = dx2
                remainder -= 1
            else:
                c = columns
                dx = dx1
            y = (i + 0.5) * dy
            for j in range(c):
                x = (j + 0.5) * dx
                coordinates.append([x, y])
        return np.array(coordinates)

    def _cal_pair_energy(self, dx, dy):
        dx = dx - np.round(dx / self.L) * self.L
        dy = dy - np.round(dy / self.L) * self.L
        r = np.sqrt(np.square(dx) + np.square(dy))

        phi = np.ones(r.shape)
        phi[r <= self.sigma0] = np.inf
        phi[r >= self.sigma1] = 0
        return phi

    def cal_total_energy(self):
        total_U = 0
        x, y = self.r_N.T
        for i in range(np.size(x) - 1):
            dx = x[i+1:] - x[i] 
            dy = y[i+1:] - y[i]
            phi = self._cal_pair_energy(dx, dy)
            total_U += np.sum(phi)
        return total_U

    def cal_delta_energy(self, i, dr):
        x, y = self.r_N.T
        x0, y0 = self.r_N[i]
        dx = x0 - x[np.arange(len(x)) != i]
        dy = y0 - y[np.arange(len(y)) != i]
        phi0 = self._cal_pair_energy(dx, dy)

        x1, y1 = self.r_N[i] + dr
        dx = x1 - x[np.arange(len(x)) != i]
        dy = y1 - y[np.arange(len(y)) != i]
        phi = self._cal_pair_energy(dx, dy)
        return np.sum(phi) - np.sum(phi0)

    def select_particle(self, i):
        return self.r_N[i]

    def move_particle(self, i, dr, dU):
        self.r_N[i] = self.r_N[i] + dr
        self.U = self.U + dU
