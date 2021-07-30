#!/usr/bin/env python

"""
Contains helper functions to generate plots.
"""

import matplotlib.ticker as ticker

from matplotlib.ticker import AutoMinorLocator


def set_axis(ax, axis="both"):
    if axis == "both":
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "{:g}".format(x))
        )
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: "{:g}".format(y))
        )
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=4)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=20)
    elif axis == "x":
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "{:g}".format(x))
        )
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=4)
        ax.tick_params(axis="x", which="both", direction="in", labelsize=20)
    elif axis == "y":
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: "{:g}".format(y))
        )
        ax.tick_params(which="major", length=8)
        ax.tick_params(which="minor", length=4)
        ax.tick_params(axis="y", which="both", direction="in", labelsize=20)
