# -*- coding: utf-8 -*-
"""
Created on 2026-09-14
@author: Ruowen Xiao
"""

# %% import dependencies
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# %% Load Data
def load_data(str_path, prefix):
    parent_dir = Path(str_path)
    data_dirs = sorted(
        parent_dir.glob(prefix + "*"),
        key=lambda path: float(path.name.removeprefix(prefix))
    )

    results = []
    for data_dir in data_dirs:
        with open(data_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        results.append({
            "energy_equi": np.load(data_dir / "energy_equi.npy"),
            "energy_prod": np.load(data_dir / "energy_prod.npy"),
            "final_config": np.load(data_dir / "final_config.npy"),
            "metadata": metadata
        })
    return results

density_results = load_data("../output/density", "rho_")
temp_results = load_data("../output/temperature", "T_")

# %% Define Common Functions: Energy, Configurations
def draw_energy(results):
    for result in results:
        U_equi = result["energy_equi"]
        U_prod = result["energy_prod"]
        T_star = result["metadata"]["T_star"]
        rho_star = result["metadata"]["rho_star"]
        sample_interval = result["metadata"]["sample_interval"]

        U = np.concatenate((U_equi, U_prod))
        steps = np.arange(U.size) * sample_interval

        fig, ax = plt.subplots()
        ax.plot(steps, U)
        ax.set_title(fr"$T^{{\star}} = {T_star}$, "
                     fr"$\rho^{{\star}} = {rho_star}$")
        plt.show()

def draw_configs(results):
    for result in results:
        config = result["final_config"]
        T_star = result["metadata"]["T_star"]
        rho_star = result["metadata"]["rho_star"]

        fig, ax = plt.subplots()
        ax.scatter(*config.T)
        ax.set_title(fr"$T^{{\star}} = {T_star}$, "
                     fr"$\rho^{{\star}} = {rho_star}$")
        plt.show()

# %%
# ================================================================================
#                               VARIOUS DENSITIES
# ================================================================================

# %% Various Densities: Energy
draw_energy(density_results)

# %% Various Densities: Configruations
draw_configs(density_results)

# %% Various Densities: Radial Distribution
# TODO
for result in density_results:
    config = result["final_config"]
    T_star = result["metadata"]["T_star"]
    rho_star = result["metadata"]["rho_star"]

# %%
# ================================================================================
#                               TEMPERATURE DROP
# ================================================================================

# %% Temperature Drop: Energy
draw_energy(temp_results)

# %% Temperature Drop: Configurations
draw_configs(temp_results)

# %% Temperature Drop: Phase Transition
x = []
y = []
for result in temp_results:
    U = result["energy_prod"]
    T_star = result["metadata"]["T_star"]
    flunctuation = np.mean(np.square(U)) - np.square(np.mean(U))
    Cv = flunctuation / T_star**2

    x.append(T_star)
    y.append(Cv)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlabel(r"Temperature, $T^{\star}$")
ax.set_ylabel(r"Specific heat, $C_V$")
ax.set_title(r"Thermodynamic properties at density $\rho^{\star}$ = 0.291")
plt.show()

# %%
