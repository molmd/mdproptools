#!/usr/bin/env python

"""
Calculates diffusion coefficient using Einstein relation from LAMMPS trajectory files.
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter

from pymatgen.io.lammps.outputs import parse_lammps_log, parse_lammps_dumps

from mdproptools.common import constants
from mdproptools.common.com_mols import calc_com
from mdproptools.utilities.plots import set_axis

__author__ = "Rasha Atwi, Matthew Bliss"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "Sep 2021"
__version__ = "0.0.1"


class Diffusion:
    def __init__(
        self, timestep=1, units="real", outputs_dir=None, diff_dir=None 
    ):
        self.units = units
        if self.units not in constants.SUPPORTED_UNITS:
            raise KeyError(
                "Unit type not supported. Supported units are: "
                + str(constants.SUPPORTED_UNITS)
            )
        self.outputs_dir = outputs_dir or os.getcwd()
        self.diff_dir = diff_dir or os.getcwd()
        self.timestep = timestep

    @staticmethod
    def _prepare_unwrapped_coords(dump):
        # Ensure access to unwrapped coordinates, may want to add scaled
        # coordinates in future ('sx', 'sxu', etc.)
        if "xu" and "yu" and "zu" not in dump.data.columns:
            assert (
                "x" and "y" and "z" in dump.data.columns
            ), "Missing wrapped and unwrapped coordinates (x y z xu yu zu)"
            assert "ix" and "iy" and "iz" in dump.data.columns, (
                "Missing unwrapped coordinates (xu yu zu) and box location ("
                "ix iy iz) for converting wrapped coordinates (x y z) into "
                "unwrapped coordinates. "
            )
            box_x_length = dump.box.bounds[0][1] - dump.box.bounds[0][0]
            box_y_length = dump.box.bounds[1][1] - dump.box.bounds[1][0]
            box_z_length = dump.box.bounds[2][1] - dump.box.bounds[2][0]
            dump.data["xu"] = dump.data["x"].add(dump.data["ix"].multiply(box_x_length))
            dump.data["yu"] = dump.data["y"].add(dump.data["iy"].multiply(box_y_length))
            dump.data["zu"] = dump.data["z"].add(dump.data["iz"].multiply(box_z_length))
        return dump

    def _calculate_type_com(self, df, ind):
        return df.groupby(ind).apply(
            lambda x: pd.Series(
                x["mass"].values @ x[["xu", "yu", "zu"]].values / x.mass.sum(),
                index=["xu", "yu", "zu"],
            )
        )

    def _modify_dump_coordinates(self, msd_df):
        ref_com = self._calculate_type_com(msd_df.xs(0, 0), ["type"])
        com = self._calculate_type_com(msd_df, ["Time (s)", "type"])
        drift = com - com[[]].join(ref_com)
        msd_df.loc[:, ["xu", "yu", "zu"]] -= drift
        return msd_df

    def detect_linear_region():
        pass

    def get_msd_from_dump(
        self,
        filename,
        tao_coeff=4,
        msd_type="com",
        num_mols=None,
        num_atoms_per_mol=None,
        mass=None,
        com_drift=False,
        avg_interval=False,
    ):
        dumps = list(parse_lammps_dumps(f"{self.outputs_dir}/{filename}"))
        msd_dfs = []
        for dump in dumps:
            assert "id" in dump.data.columns, "Missing atom id's in dump file."
            dump.data = dump.data.sort_values(by=["id"])
            dump.data.reset_index(inplace=True)
            dump = self._prepare_unwrapped_coords(dump)
            if msd_type == "allatom":
                df = dump.data[["id", "xu", "yu", "zu"]]
                msd_dfs.append(df)
                id_cols = ["id"]
                col_1d = ["Time (s)"]
            elif msd_type == "com":
                df = calc_com(
                    dump,
                    num_mols,
                    num_atoms_per_mol,
                    mass,
                    atom_attributes=["xu", "yu", "zu"],
                )
                # convert to SI units
                df["mass"] = df["mass"] * constants.MASS_CONVERSION[self.units]
                df.reset_index(inplace=True)
                msd_dfs.append(df)
                id_cols = ["type", "mol_id"]
                col_1d = ["Time (s)", "type"]
            # convert to SI units
            df.loc[:, "xu"] = df["xu"] * constants.DISTANCE_CONVERSION[self.units]
            df.loc[:, "yu"] = df["yu"] * constants.DISTANCE_CONVERSION[self.units]
            df.loc[:, "zu"] = df["zu"] * constants.DISTANCE_CONVERSION[self.units]
            df["Time (s)"] = (
                dump.timestep * self.timestep * constants.TIME_CONVERSION[self.units]
            )
        msd_df = pd.concat(msd_dfs).set_index(["Time (s)"] + id_cols).sort_index()
        if msd_type == "com" and com_drift:
            msd_df = self._modify_dump_coordinates(msd_df)
        coords = ["xu", "yu", "zu"]
        disps = ["dx2", "dy2", "dz2"]
        msd_all = msd_df.copy()
        ref_df = msd_all.xs(0, 0)
        msd_all[disps] = (msd_all[coords] - msd_all[[]].join(ref_df[coords])) ** 2
        msd_all["msd"] = msd_all[disps].sum(axis=1)
        one_d_cols = disps + ["msd"]
        msd_all = msd_all[one_d_cols]
        msd = msd_all.groupby(col_1d).mean()
        if msd_type == "com":
            msd = msd.reset_index().pivot(columns="type", index="Time (s)")
            msd = msd.sort_index(axis=1, level=1)
            msd.columns = ["".join([str(i) for i in v]) for v in msd.columns.values]
        msd.reset_index(inplace=True)
        msd_all.reset_index(inplace=True)
        if avg_interval:
            times = list(msd_df.index.levels[0])
            times = times[::tao_coeff]
            msd_df = msd_df[msd_df.index.get_level_values(0).isin(times)]
            msd_df[disps] = (
                msd_df[coords]
                .groupby(id_cols)
                .transform(lambda x: (x - x.shift(1)) ** 2)
            )
            msd_df.drop(0, level=0)
            msd_df["msd"] = msd_df[disps].sum(axis=1)
            msd_df = msd_df[one_d_cols]
            msd_int = msd_df.groupby(id_cols).mean().reset_index()
            return msd, msd_all, msd_int
        return msd, msd_all

    def get_msd_from_log(self, log_pattern):
        log_files = glob.glob(f"{self.outputs_dir}/{log_pattern}")
        # based on pymatgen.io.lammps.outputs.parse_lammps_dumps function
        if len(log_files) > 1:
            pattern = r"%s" % log_pattern.replace("*", "([0-9]+)")
            pattern = ".*" + pattern.replace("\\", "\\\\")
            files = sorted(files, key=lambda f: int(re.match(pattern, f).group(1)))
        list_log_df = []
        for file in log_files:
            log_df = parse_lammps_log(file)
            list_log_df.append(log_df[0])
        for p, l in enumerate(list_log_df[:-1]):
            list_log_df[p] = l[:-1]
        full_log = pd.concat(list_log_df, ignore_index=True)
        msd = full_log.filter(regex="msd")
        for col in msd:
            msd.loc[:, col] = msd[col] * constants.DISTANCE_CONVERSION[self.units] ** 2
        msd["Time (s)"] = (
            full_log["Step"] * self.timestep * constants.TIME_CONVERSION[self.units]
        )
        return msd

    def calc_diff(
        self,
        msd,
        initial_time=None,
        final_time=None,
        dimension=3,
        diff_names=None,
        save=False,
        plot=False,
    ):
        # initial and final time in seconds
        if initial_time is None:
            initial_time = {}
        if final_time is None:
            final_time = {}
        min_t = min(msd["Time (s)"])
        max_t = max(msd["Time (s)"])
        msd_col_names = [col for col in msd.columns if "msd" in col.lower()]
        diff = np.zeros((len(msd_col_names), 3))
        models = []
        counter = 0
        for ind, col in enumerate(msd_col_names):
            msd_df = msd[
                (msd["Time (s)"] >= initial_time.get(counter, min_t))
                & (msd["Time (s)"] <= final_time.get(counter, max_t))
            ]
            counter += 1
            model = sm.OLS(msd_df[col], msd_df["Time (s)"]).fit()
            models.append(model)
            diff[ind] = [
                model.params[0] / (2 * dimension),
                model.bse[0] / (2 * dimension),
                model.rsquared,
            ]
            if save:
                if diff_names:
                    diff_file = f"{self.diff_dir}/diff_{diff_names[ind]}.txt"
                else:
                    diff_file = f"{self.diff_dir}/diff_{ind + 1}.txt"
                with open(diff_file, "w") as f:
                    f.write(str(model.summary()))
        ind = diff_names or [i + 1 for i in range(len(msd_col_names))]
        diffusion = pd.DataFrame(
            diff, columns=["diffusion (m2/s)", "std", "R2"], index=ind,
        )

        if plot:
            paired = plt.get_cmap("Paired")
            colors = iter(paired(np.linspace(0, 1, 10)))
            ncols = 2
            nrows = int(np.ceil(len(msd_col_names) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
            fig_log, axes_log = plt.subplots(nrows, ncols, figsize=(12, 8))
            time_data = msd["Time (s)"] * 10 ** 9
            for i, (ax, ax_log, col) in enumerate(
                zip(axes.flatten(), axes_log.flatten(), msd_col_names)
            ):
                color = next(colors)
                y_formatter = ScalarFormatter(useOffset=False)

                # normal plot
                pred = models[i].predict()
                ax.plot(
                    time_data, msd[col], color=color, linewidth=2, label=ind[i],
                )
                ax.plot(time_data, pred, color="k", ls="--", linewidth=2)
                ax.locator_params(axis="y", nbins=6)
                # log plot
                st_line = 10 ** (np.log10(msd[col].max()) - np.log10(time_data.max()))
                ax_log.plot(
                    time_data, msd[col], color=color, linewidth=2, label=ind[i],
                )
                ax_log.plot(
                    time_data, time_data * st_line, color="k", ls="--", linewidth=2
                )
                ax_log.set(xscale="log", yscale="log")

                for axis in [ax, ax_log]:
                    set_axis(axis, axis="both")
                    axis.legend(
                        fontsize=16, frameon=False,
                    )
                    axis.set_xlabel(r"$\mathrm{Time, 10^9 (s)}$", fontsize=18)
                    axis.set_ylabel(r"$\mathrm{MSD\ (m^2)}$", fontsize=18)
                    axis.yaxis.set_major_formatter(y_formatter)
                    axis.yaxis.offsetText.set_fontsize(18)

            for figure, axis, name in zip(
                [fig, fig_log], [axes, axes_log], ["msd.png", "msd_log.png"]
            ):
                if len(msd_col_names) % 2 != 0:
                    figure.delaxes(ax=axis.flatten()[-1])
                figure.tight_layout()
                figure.savefig(
                    f"{self.diff_dir}/{name}", bbox_inches="tight", pad_inches=0.1,
                )

        diffusion.to_csv(f"{self.diff_dir}/diffusion.csv")
        print("Diffusion results written to a .csv file.")
        return diffusion

    def get_diff_dist(
        self, msd_int, dump_freq, dimension=3, tao_coeff=4, plot=False, diff_names=None
    ):
        delta = dump_freq * self.timestep * constants.TIME_CONVERSION[self.units]
        msd_int["diff"] = msd_int["msd"] / (2 * dimension * tao_coeff * delta)
        if plot:
            paired = plt.get_cmap("Paired")
            colors = iter(paired(np.linspace(0, 1, 10)))
            if "type" in msd_int.columns:
                # if data is available for the com of each molecule type
                groups = msd_int.groupby("type")
                ind = diff_names or [i + 1 for i in range(len(groups))]
                ncols = 2
                nrows = int(np.ceil(groups.ngroups / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
                for ax, (key, grp) in zip(axes.flatten(), groups):
                    color = next(colors)
                    set_axis(ax, axis="both")
                    sns.histplot(
                        grp["diff"] * 10 ** 9,
                        bins="sqrt",
                        color=color,
                        edgecolor="k",
                        label=ind[key - 1],
                        kde=True,
                        stat="density",
                        ax=ax,
                    )
                    ax.legend(
                        fontsize=16, frameon=False,
                    )
                    ax.set_xlabel(
                        r"$\mathrm{Diffusivity, 10^{-9}\ (m^2/s)}$", fontsize=18
                    )
                    ax.set_ylabel(
                        "Frequency", fontsize=18,
                    )
                    ax.xaxis.get_major_ticks()[1].label1.set_visible(False)
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    y_formatter = ScalarFormatter(useOffset=False)
                    ax.yaxis.set_major_formatter(y_formatter)
                    ax.yaxis.offsetText.set_fontsize(18)
                    ax.locator_params(axis="y", nbins=6)
                if len(groups) % 2 != 0:
                    fig.delaxes(ax=axes.flatten()[-1])
                fig.tight_layout()
                fig.savefig(
                    f"{self.diff_dir}/diff_dist.png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
            else:
                # if data is available for all atoms
                fig, ax = plt.subplots(figsize=(8, 6))
                set_axis(ax, axis="both")
                sns.histplot(
                    msd_int["diff"] * 10 ** 9,
                    bins="sqrt",
                    color=next(colors),
                    edgecolor="k",
                    kde=True,
                    stat="density",
                    ax=ax,
                )
                ax.set_xlabel(r"$\mathrm{Diffusivity, 10^{-9}\ (m^2/s)}$", fontsize=18)
                ax.set_ylabel(
                    "Frequency", fontsize=18,
                )
                ax.xaxis.get_major_ticks()[1].label1.set_visible(False)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                y_formatter = ScalarFormatter(useOffset=False)
                ax.yaxis.set_major_formatter(y_formatter)
                ax.yaxis.offsetText.set_fontsize(18)
                ax.locator_params(axis="y", nbins=6)
                fig.tight_layout()
                fig.savefig(
                    f"{self.diff_dir}/diff_dist.png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
