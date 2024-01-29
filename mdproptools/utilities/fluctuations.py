import os

import numpy as np
import matplotlib.pyplot as plt

from mdproptools.common import constants
from mdproptools.utilities.plots import set_axis


def _get_stats(stats):
    return "(" + ", ".join([f"{k}:{v: .4g}" for k, v in stats.items()]) + ")"


def plot_fluctuations(
    log, log_prop, title, filename, timestep=1, units="real", working_dir=None
):
    """
    Plot fluctuations of a specified property from a LAMMPS log file over time and
    print the mean and standard deviation of the property to the console.

    Args:
        log (DataFrame): Pandas DataFrame containing the log data.
        log_prop (str): The property within the log to plot. Should match the property
            name in the log.
        title (str): The title of the plot.
        filename (str): Name of the file to save the plot to.
        timestep (float, optional): Timestep used in the simulations in the same units
            specified as input; defaults to 1 fs when real units are used.
        units (str, optional): Units used in the LAMMMPS simulations; used to convert to
            SI units; defaults to real unit.
        working_dir (str, optional): The working directory to save the plot in.
            If None, the current working directory is used; defaults to None.

    Returns:
        None
    """
    working_dir = working_dir or os.getcwd()
    fig, ax = plt.subplots(figsize=(8, 6), sharey=False)
    set_axis(ax, axis="both")
    time_data = log["Step"] * timestep * constants.TIME_CONVERSION[units] * 10**9
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
    return stats["mean"], stats["std"]
