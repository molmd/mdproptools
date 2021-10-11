import os
import re
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy import optimize, integrate
from matplotlib.ticker import ScalarFormatter

from pymatgen.io.lammps.outputs import parse_lammps_log

from mdproptools.common import constants
from mdproptools.utilities.plots import set_axis

sns.set()
sns.set_style("ticks", {"xtick.major.size": 0.1, "ytick.major.size": 0.1})
sns.set_palette(sns.color_palette(["#00807D", "#8CC8D5", "#A7A7A7"], n_colors=3))


def _get_stats(stats):
    return "(" + ", ".join([f"{k}:{v: .4g}" for k, v in stats.items()]) + ")"


def plot_fluctuations(
    log, log_prop, title, filename, timestep=1, units="real", working_dir=None
):
    working_dir = working_dir or os.getcwd()
    fig, ax = plt.subplots(figsize=(8, 6), sharey=False)
    set_axis(ax, axis="both")
    time_data = log["Step"] * timestep * constants.TIME_CONVERSION[units] * 10 ** 9
    stats = log[log_prop].describe().loc[["mean", "std"]].to_dict()
    print("{}: mean = {}, std = {}".format(log_prop, stats["mean"], stats["std"]))
    ax.plot(time_data, log[log_prop], linewidth=2, color="red")
    ax.axhline(np.mean(log[log_prop]), linewidth=2, color="#000000", ls="--")

    ax.set_title("{} {}".format(title, _get_stats(stats)), fontsize=18)
    ax.set_xlabel(r"$\mathrm{Time, 10^9 (m^2/s)}$", fontsize=18)
    ax.set_xlim(0, None)
    ax.set_ylim(
        log[log_prop].min() * 2 if log[log_prop].min() < 0 else log[log_prop].min() / 2,
        log[log_prop].max() * 2
        if log[log_prop].max() > 0
        else -log[log_prop].max() * 2,
    )
    fig.tight_layout(pad=3)
    fig.savefig(f"{working_dir}/{filename}", bbox_inches="tight", pad_inches=0.1)
