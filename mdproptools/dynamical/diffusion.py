#!/usr/bin/env python

"""
Calculates diffusion coefficient using Einstein relation from LAMMPS trajectory files.
"""

import os, re, glob

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter

from pymatgen.io.lammps.outputs import parse_lammps_log, parse_lammps_dumps

from mdproptools.common import constants
from mdproptools.utilities.log import concat_log
from mdproptools.common.com_mols import calc_com
from mdproptools.utilities.plots import set_axis

__author__ = "Rasha Atwi, Matthew Bliss"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "Sep 2021"
__version__ = "0.0.6"


class Diffusion:
    """
    Class to calculate diffusion coefficients based on the mean square displacement (msd)
    from LAMMPS trajectory or log files using Einstein expression. Supports calculation
    of msd for all atoms or center of mass of molecules.
    """
    def __init__(self, timestep=1, units="real", outputs_dir=None, diff_dir=None):
        """
        Creates an instance of the Diffusion class.

        Args:
            timestep (int or float): Timestep used in the LAMMPS simulation in the same
                units specified as input; default is 1 fs for real units.
            units (str): Units used in the LAMMMPS simulations; used to convert to SI
                units; defaults to real unit.
            outputs_dir (str): Directory where the LAMMPS trajectory or log files are
                located; defaults to current working directory.
            diff_dir (str): Directory where the diffusion results (.txt, .csv, and .png
                files) will be saved; defaults to current working directory.
        """
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
        msd_type="com",
        num_mols=None,
        num_atoms_per_mol=None,
        mass=None,
        com_drift=False,
        avg_interval=False,
        tao_coeff=4,
    ):
        """
        Calculate the mean square displacement (MSD) from a LAMMPS trajectory file. MSD
        is calculated as the sum of the square of the displacement in each direction
        (dx, dy, dz), averaged over all atoms or the center of mass of species of each
        specie type in the system. The first point in time is taken as the reference for
        calculating the displacements at each time. Note that this is not the case
        for the msd_int output returned when `avg_interval` is True, where the reference
        for each time interval is the previous time interval, allowing to average over
        all possible choices of the time interval over the entire trajectory,
        thereby making it possible to get a linear MSD as a function of time for each
        atom or specie in the system.

        Args:
            filename (str): Pattern of the name of the LAMMPS dump files.
            msd_type (str, optional): Type of MSD to calculate. Options are 'allatom'
                or 'com' for the center of mass of each specie type in the system;
                defaults to com.
            num_mols (list of int, optional): Number of molecules of each molecule type;
                should be consistent with PackmolRunner input; required if `msd_type`
                is set to com; defaults to None.
            num_atoms_per_mol (list of int, optional): Number of atoms in each molecule
                type; order is similar to that in num_mols; required if `msd_type` is
                set to com; defaults to None.
            mass (list of float, optional): Mass of unique atom types in a LAMMPS dump
                file; should be in the same order as in the LAMMPS data file and in
                the same units specified as input, e.g. if the "real" units are used,
                the masses should be in g/mole; required if `msd_type` is set to com and
                the masses are not available in the dump file; defaults to None.
            com_drift (bool, optional): Whether to correct for the center of mass drift
                in the system; only used when `msd_type` is com; defaults to False.
            avg_interval (bool, optional): Whether to calculate the msd for individual
                atoms or individual species from each type; can be later used to
                calculate the distribution of diffusion coefficients for a given atom or
                specie rather than only having the average diffusion coefficient per
                atom or specie type; defaults to False.
            tao_coeff (int, optional): Time interval (step, unitless) to use when
                sampling the trajectory to get msd_int, which corresponds to the msd
                for each atom (when `msd_type` is allatom) or specie (when `msd_type`
                is com), averaged over all possible choices of time interval over the
                entire trajectory; for example, if your dump frequency is every
                50,000 steps, and you choose tao to be 4, then the time interval for
                calculating the msd will be every 200,000 steps; defaults to 4.

        Returns:
            tuple of pd.DataFrames:
                - msd: The square of the displacement in each direction along with the
                    MSD as a function of time, averaged over ALL atoms (when `msd_type`
                    is allatom) or ALL species of the same type (when `msd_type` is com).
                    The first time step is the reference.
                - msd_all: The square of the displacement in each direction along with
                    the MSD as a function of time for EACH atom (when `msd_type` is
                    allatom) or EACH specie (when `msd_type` is com). The first time
                    step is the reference. Not that this is used for getting the msd
                    dataframe.
                - msd_int: The square of the displacement in each direction along with
                    the MSD for EACH atom or EACH specie, averaged over all possible
                    choices of time interval over the entire trajectory; only returned
                    if `avg_interval` is set to True. The reference for each time
                    interval is the previous time interval.
        """
        dumps = parse_lammps_dumps(f"{self.outputs_dir}/{filename}")
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
            else:
                raise ValueError("msd_type must be 'allatom' or 'com'.")
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
        """
        Extract the MSD data from a LAMMPS log file(s) and convert to SI units. The
        log file(s) should include one or more columns with 'msd' in their names to
        identify the MSD data.

        Args:
            log_pattern (str): A glob pattern string to identify the LAMMPS log files.
                The pattern should match all relevant log files in the `outputs_dir`
                directory of the simulation outputs.

        Returns:
            pd.DataFrame: A DataFrame containing the MSD data as a function of
                simulation time. Each MSD column from the original log files is
                preserved and converted to meters squared (m^2). A new column,
                'Time (s)', is added to represent the simulation time in seconds.
        """
        full_log = concat_log(log_pattern, step=None, working_dir=self.outputs_dir)
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
        """
        Calculate the diffusion coefficient from the MSD data using the Einstein
        relation. The diffusion coefficient is calculated as the slope of the linear
        fit of the MSD as a function of time. The standard error and the R-squared value
        are also calculated. The results are saved to a .csv file and optionally plotted.
        Note that if both initial and final time are not provided, the entire MSD data
        will be used to calculate the diffusion coefficients.

        Args:
            msd (pd.DataFrame): DataFrame containing the MSD data as a function of time.
            initial_time (dict, optional): Initial time in seconds for each MSD column;
                defaults to None.
            final_time (dict, optional): Final time in seconds for each MSD column;
                defaults to None.
            dimension (int, optional): Dimension of the system; defaults to 3.
            diff_names (list, optional): List of names for the diffusion coefficients
                (e.g. if MSD data is for com of each molecule type, the names can be
                the molecule names); defaults to None in which case the names are
                set to the column numbers.
            save (bool, optional): Whether to save the ordinary least squares model
                results from statsmodel to a .txt file; defaults to False.
            plot (bool, optional): Whether to plot the MSD and log MSD as a function of
                time along with the fitted model; defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the diffusion coefficients, the standard
                error, and the R-squared value for each MSD column. The results are
                also saved to a diffusion.csv file in the `diff_dir` directory.
        """
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
            diff,
            columns=["diffusion (m2/s)", "std", "R2"],
            index=ind,
        )

        if plot:
            paired = plt.get_cmap("Paired")
            colors = iter(paired(np.linspace(0, 1, 10)))
            ncols = 2
            nrows = int(np.ceil(len(msd_col_names) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
            fig_log, axes_log = plt.subplots(nrows, ncols, figsize=(12, 8))
            time_data = msd["Time (s)"] * 10**9
            for i, (ax, ax_log, col) in enumerate(
                zip(axes.flatten(), axes_log.flatten(), msd_col_names)
            ):
                color = next(colors)
                y_formatter = ScalarFormatter(useOffset=False)

                # normal plot
                pred = models[i].predict()
                ax.plot(
                    time_data,
                    msd[col],
                    color=color,
                    linewidth=2,
                    label=ind[i],
                )
                ax.plot(time_data, pred, color="k", ls="--", linewidth=2)
                ax.locator_params(axis="y", nbins=6)
                # log plot
                st_line = 10 ** (np.log10(msd[col].max()) - np.log10(time_data.max()))
                ax_log.plot(
                    time_data,
                    msd[col],
                    color=color,
                    linewidth=2,
                    label=ind[i],
                )
                ax_log.plot(
                    time_data, time_data * st_line, color="k", ls="--", linewidth=2
                )
                ax_log.set(xscale="log", yscale="log")

                for axis in [ax, ax_log]:
                    set_axis(axis, axis="both")
                    axis.legend(
                        fontsize=16,
                        frameon=False,
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
                    f"{self.diff_dir}/{name}",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )

        diffusion.to_csv(f"{self.diff_dir}/diffusion.csv")
        print("Diffusion results written to a .csv file.")
        return diffusion

    def get_diff_dist(
        self, msd_int, dump_freq, dimension=3, tao_coeff=4, plot=False, diff_names=None
    ):
        """
        Calculate the distribution of diffusion coefficients from the MSD data (msd_int)
        obtained when `avg_interval` is set to True in `get_msd_from_dump`. Plot the
        distribution of diffusion coefficients for the atoms or each species type in the
        system and save the results to a .csv file.

        Args:
            msd_int (pd.DataFrame): DataFrame containing the MSD data for EACH atom or
                EACH specie. See `get_msd_from_dump`.
            dump_freq (int): Frequency of the LAMMPS dump files in steps.
            dimension (int, optional): Dimension of the system; defaults to 3.
            tao_coeff (int, optional): Time interval (step, unitless) used when
                sampling the trajectory to get msd_int; defaults to 4.
            plot (bool, optional): Whether to plot the distribution of diffusion
                coefficients; defaults to False.
            diff_names (list, optional): List of names for the diffusion coefficients
                (e.g. if MSD data is for com of each molecule type, the names can be
                the molecule names); defaults to None in which case the names are
                set to the column numbers.

        Returns:
            None
        """
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
                        grp["diff"] * 10**9,
                        bins="sqrt",
                        color=color,
                        edgecolor="k",
                        label=ind[key - 1],
                        kde=True,
                        stat="density",
                        ax=ax,
                    )
                    ax.legend(
                        fontsize=16,
                        frameon=False,
                    )
                    ax.set_xlabel(
                        r"$\mathrm{Diffusivity, 10^{-9}\ (m^2/s)}$", fontsize=18
                    )
                    ax.set_ylabel(
                        "Frequency",
                        fontsize=18,
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
                    msd_int["diff"] * 10**9,
                    bins="sqrt",
                    color=next(colors),
                    edgecolor="k",
                    kde=True,
                    stat="density",
                    ax=ax,
                )
                ax.set_xlabel(r"$\mathrm{Diffusivity, 10^{-9}\ (m^2/s)}$", fontsize=18)
                ax.set_ylabel(
                    "Frequency",
                    fontsize=18,
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
        return msd_int

