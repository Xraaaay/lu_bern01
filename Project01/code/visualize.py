# -*- coding: utf-8 -*-
"""
Created on 2026-09-14
@author: Ruowen Xiao
"""

# %% import dependencies
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np
import json

from IPython.display import HTML
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
            "energy": np.load(data_dir / "energy.npy"),
            "configs": np.load(data_dir / "configs.npy"),
            "final_config": np.load(data_dir / "final_config.npy"),
            "metadata": metadata
        })
    return results

density_results = load_data("../output/density", "rho_")
temp_results = load_data("../output/temperature", "T_")

# %% Define Common Functions: Energy, Configurations
def draw_energy(results, path_template):
    for result in results:
        U = result["energy"]
        T_star = result["metadata"]["T_star"]
        rho_star = result["metadata"]["rho_star"]
        n_prod_runs = result["metadata"]["n_production_runs"]
        sample_interval = result["metadata"]["sample_interval_energy"]

        steps = np.arange(U.size) * sample_interval
        prod_start = U.size * sample_interval - n_prod_runs

        fig, ax = plt.subplots()
        ax.plot(steps, U)
        ax.axvline(x=prod_start, linestyle="--")
        ax.set_title(fr"$T^{{\star}} = {T_star}$, "
                     fr"$\rho^{{\star}} = {rho_star}$")

        path = path_template.format(rho_star=rho_star, T_star=T_star)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)

        plt.show()

def draw_configs(results, path_template):
    for result in results:
        config = result["final_config"]
        T_star = result["metadata"]["T_star"]
        rho_star = result["metadata"]["rho_star"]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(*config.T)
        ax.set_title(fr"$T^{{\star}} = {T_star}$, "
                     fr"$\rho^{{\star}} = {rho_star}$")
        ax.set_box_aspect(1)

        path = path_template.format(rho_star=rho_star, T_star=T_star)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)

        plt.show()

def draw_animation(results, path_template):
    for result in results:
        configs = result["configs"]
        N = result["metadata"]["N"]
        rho_star = result["metadata"]["rho_star"]
        T_star = result["metadata"]["T_star"]
        L = math.sqrt(N / rho_star)

        fig, ax = plt.subplots()
        scatter = ax.scatter(configs[0, :, 0], 
                            configs[0, :, 1], 
                            s=50)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")

        def update(i):
            scatter.set_offsets(configs[i])
            return scatter,

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(configs),
            interval=50,
            blit=True
        )

        path = path_template.format(rho_star=rho_star, T_star=T_star)
        ani.save(path, writer="pillow", fps=20)

        # plt.close(fig)
        # HTML(ani.to_jshtml())

def radial_dist(configs: np.ndarray, L, dr=0.1):
    n_configs, n_particles, _ = configs.shape
    r_max = L / 2
    bins = np.arange(0, r_max + dr, dr)
    counts = np.zeros(len(bins) - 1)

    i, j = np.triu_indices(n_particles, k=1)  # i < j
    for config in configs:
        delta = config[:, None, :] - config[None, :, :]
        delta -= L * np.round(delta / L)  # PBC

        distances = np.linalg.norm(delta, axis=-1)
        pair_distances = distances[i, j]

        hist, _ = np.histogram(pair_distances, bins=bins)
        counts += hist

    shell_area = np.pi * (np.square(bins[1:]) - np.square(bins[:-1]))
    expected_counts = (
        n_configs 
        * n_particles * (n_particles - 1) / 2 
        * shell_area / L**2
    )

    g_r = counts / expected_counts
    r = (bins[1:] + bins[:-1]) / 2

    return r, g_r


# %%
# ================================================================================
#                               VARIOUS DENSITIES
# ================================================================================

# %% Various Densities: Energy
output_path = "../results/density/rho_{rho_star}/energy_rho_{rho_star}.png"
draw_energy(density_results, output_path)

# %% Various Densities: Configruations
output_path = "../results/density/rho_{rho_star}/config_rho_{rho_star}.png"
draw_configs(density_results, output_path)

# %% Various Densities: Radial Distribution
for result in density_results:
    configs = result["configs"]
    N = result["metadata"]["N"]
    T_star = result["metadata"]["T_star"]
    rho_star = result["metadata"]["rho_star"]
    n_prod_runs = result["metadata"]["n_production_runs"]
    sample_interval = result["metadata"]["sample_interval_energy"]

    L = math.sqrt(N / rho_star)
    sample_count = n_prod_runs // sample_interval
    prod_configs = configs[-sample_count:]

    r, g_r = radial_dist(prod_configs, L)

    fig, ax = plt.subplots()
    ax.plot(r, g_r)
    ax.axvline(1, linestyle="--", color="gray")
    ax.axvline(2.5, linestyle="--", color="gray")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g(r)$")

    output_path = f"../results/density/rho_{rho_star}/gr_rho_{rho_star}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

# %% Draw animation
output_path = "../results/density/rho_{rho_star}/config_rho_{rho_star}.gif"
draw_animation(density_results, output_path)

# %%
# ================================================================================
#                               TEMPERATURE DROP
# ================================================================================

# %% Temperature Drop: Energy
output_path = "../results/temperature/T_{T_star}/energy_T_{T_star}.png"
draw_energy(temp_results, output_path)

# %% Temperature Drop: Configurations
output_path = "../results/temperature/T_{T_star}/config_T_{T_star}.png"
draw_configs(temp_results, output_path)

# %% Temperature Drop: Phase Transition
x = []
y = []
for result in temp_results:
    U = result["energy"]
    T_star = result["metadata"]["T_star"]
    n_prod_runs = result["metadata"]["n_production_runs"]
    sample_interval = result["metadata"]["sample_interval_energy"]

    sample_count = n_prod_runs // sample_interval
    U_prod = U[-sample_count:]

    flunctuation = np.mean(np.square(U_prod)) - np.square(np.mean(U_prod))
    Cv = flunctuation / T_star**2

    x.append(T_star)
    y.append(Cv)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.grid()
ax.set_xlabel(r"Temperature, $T^{\star}$")
ax.set_ylabel(r"Specific heat, $C_V$")
# ax.set_title(r"Thermodynamic properties at density $\rho^{\star}$ = 0.291")

output_path = "../results/temperature/Cv.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)

plt.show()

# %% Temperature Drop: Expected Energy
x = []
y = []
for result in temp_results:
    U = result["energy"]
    T_star = result["metadata"]["T_star"]
    n_prod_runs = result["metadata"]["n_production_runs"]
    sample_interval = result["metadata"]["sample_interval_energy"]

    sample_count = n_prod_runs // sample_interval
    U_prod = U[-sample_count:]

    x.append(T_star)
    y.append(np.mean(U_prod))

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.grid()
ax.set_xlabel(r"Temperature, $T^{\star}$")
ax.set_ylabel(r"$U$")

output_path = "../results/temperature/energy.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)

plt.show()

# %% Draw animation
output_path = "../results/temperature/T_{T_star}/config_T_{T_star}.gif"
draw_animation(temp_results, output_path)

# %%
