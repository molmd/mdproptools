#!/usr/bin/env python

"""
Calculates the ionic conductivity from LAMMPS trajectory files.
"""

import os

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import cumtrapz
from matplotlib.ticker import ScalarFormatter

from pymatgen.io.lammps.outputs import parse_lammps_dumps

from mdproptools.common import constants
from mdproptools.common.com_mols import calc_com
from mdproptools.utilities.plots import set_axis
from mdproptools.dynamical._conductivity import conductivity_loop

__author__ = "Rasha Atwi"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "Sep 2021"
__version__ = "0.0.1"


# TODO: Implement and test other methods for calculating conductivity (eg. Einstein
#  relation, nernst)


class Conductivity:
    """
    Class to calculate the ionic conductivity from MD simulations. Uses Green-Kubo
    correlation according to 10.1063/1.4890741. Computes the total ionic conductivity
    as well as the contribution of molecules types using the following steps: (1)
    calculates the center of mass of each molecule, (2) calculates the total charge flux
    and the charge flux for each molecule type, (3) calculates the charge flux
    correlation function, (4) integrates the charge flux correlation function, (5)
    detects the range where the charge flux for each molecule type is near zero and
    computes the average of the integral over this time range (this is done because
    the long-time values of the integral are subject to noise and should be disgarded
    in the ionic conductivity calculation).
    """

    def __init__(
        self,
        filename,
        num_mols,
        num_atoms_per_mol,
        mass=None,
        temp=298.15,
        timestep=1,
        units="real",
        working_dir=None,
    ):
        """
        Creates a Conductivity object.

        Args:
            filename (str): Pattern of the name of the LAMMPS dump files
            num_mols (list): Number of molecules of each molecule type; should
                be consistent with PackmolRunner input
            num_atoms_per_mol (list): Number of atoms in each molecule type; order is
                similar to that in num_mols
            mass (list): Mass of unique atom types in a LAMMPS dump file; should be in
                the same order as in the LAMMPS data file and in the same units
                specified as input, e.g. if the "real" units are used, the masses
                should be in g/mole; required if the masses are not available in the
                dump file
            temp (flaot): Temperature (K); defaults to 298.15 K
            timestep (int or float): Timestep used in the simulations in the same units
                specified as input; defaults to 1 fs when real units are used
            units (str): Units used in the LAMMMPS simulations; used to convert to SI
                units; defaults to real unit
            working_dir (str): Path of the LAMMPS dump files
        """
        self.working_dir = working_dir or os.getcwd()
        self.dumps = list(parse_lammps_dumps(f"{self.working_dir}/{filename}"))
        self.mass = mass
        self.num_mols = num_mols
        self.num_atoms_per_mol = num_atoms_per_mol
        self.units = units
        box_lengths = self.dumps[0].box.to_lattice().lengths
        self.volume = (
            np.prod(box_lengths) * constants.DISTANCE_CONVERSION[self.units] ** 3
        )  # volume in m^3
        self.temp = temp
        self.timestep = timestep
        self.time = []  # time data used to calculate GK integral

    @staticmethod
    def correlate(a, b):
        """
        Calculates the correlation function between a and b using fast Fourier
        transforms; Credits: pylat/src/calcCond.py.

        Args:
            a, b (array-like): Data to be correlated

        Returns:
            Correlated data
        """
        al = np.concatenate((a, np.zeros(len(a))), axis=0)
        bl = np.concatenate((b, np.zeros(len(b))), axis=0)
        c = np.fft.ifft(np.fft.fft(al) * np.conjugate(np.fft.fft(bl))).real
        d = c[: len(c) // 2]
        d /= (np.arange(len(d)) + 1)[::-1]
        return d

    @staticmethod
    def detect_time_range(flux, tol):
        """
        Detects the time range to use in calculating the ionic conductivity as the time
        where the charge flux correlation function is near zero; identified by
        discretizing the data, calculating the std for each group, and identifying the
        biggest group where the std is less than some tolerance value

        Args:
            flux (array-like): Charge flux correlation function
            tol (float): Tolerance value

        Returns:
            List of the start and end indexes in the data where the function is near zero
        """
        flux = pd.Series(flux, name="flux")
        time_step = max(int(len(flux) / 10000), 5)
        ind = [i // time_step for i in range(len(flux))]
        flux_groupby = flux.groupby(ind)
        flux_std = flux_groupby.transform("std")
        std = flux_std.std()
        div = std if std else 1 # to avoid dividing by zero
        flux_std = flux_std/div
        flux_std = (flux_std < tol).astype("int").to_frame()
        flux_std = (
            flux_std.rolling(
                window=4 * time_step + 1, min_periods=3 * time_step + 1, center=True
            )
            .median()
            .fillna(0)["flux"]
            .to_list()
        )
        s_e_list = []
        found_start = False
        for k, v in enumerate(flux_std):
            if v == 1 and not found_start:
                s_e_list.append((k,))
                found_start = True
            elif v < 1 and found_start:
                s_e_list[-1] = s_e_list[-1] + (k,)
                found_start = False
        if s_e_list and len(s_e_list[-1]) == 1:
            s_e_list[-1] = s_e_list[-1] + (len(flux_std) - 1,)
        max_s_e = 0
        max_s_e_ind = None
        for s_e_ind, s_e in enumerate(s_e_list):
            if s_e[1] - s_e[0] > max_s_e:
                max_s_e = s_e[1] - s_e[0]
                max_s_e_ind = s_e_ind
        return s_e_list[max_s_e_ind]

    def get_charge_flux(self):
        """
        Calculates the center of mass of each molecule and computes the charge fluxes
        for each molecule type using the molecular charges and the 3D velocity

        Returns:
            j (array-like): Charge fluxes of shape (3, # molecule types, # time steps)
        """
        inputs = []
        j = np.zeros((3, len(self.num_mols), len(self.dumps)))
        for ind, dump in enumerate(self.dumps):
            inputs.append(
                (
                    dump,
                    self.num_mols,
                    self.num_atoms_per_mol,
                    self.mass,
                    ind,
                    self.units,
                )
            )

        p = Pool(cpu_count() - 1)
        res = p.starmap(conductivity_loop, inputs)
        for i in res:
            self.time.append(i[0] * self.timestep)
            j[:, :, i[1]] = i[2]
        return j

    def correlate_charge_flux(self, flux):
        """
        Calculates the charge flux correlation function for each molecule type.

        Args:
            flux (array-like): Charge flux for each molecule type

        Returns:
            tot_flux (array-like):Charge flux correlation function for each molecule type
        """
        tot_flux = np.zeros((len(self.num_mols) + 1, flux.shape[2]))
        for i in range(len(self.num_mols)):
            for j in range(len(self.num_mols)):
                for k in range(flux.shape[0]):
                    corr = self.correlate(flux[k, i, :], flux[k, j, :])
                    tot_flux[i, :] += corr
                    tot_flux[-1, :] += corr
        return tot_flux

    def integrate_charge_flux_correlation(self, tot_flux):
        """
        Calculates the integral of the charge flux correlation function for each
        molecule type.

        Args:
            tot_flux (array-like): Charge flux correlation function for each molecule type

        Returns:
            integral (array-like): Integral of the charge flux correlation function of shape
            (3, # molecule types, # steps)
        """
        integral = np.zeros((len(tot_flux), len(tot_flux[0])))
        delta = self.time[1] - self.time[0]
        for i in range(0, len(tot_flux)):
            integral[i][1:] = cumtrapz(tot_flux[i], dx=delta)
        return integral

    def fit_curve(self, tot_flux, integral, tol):
        """
        Computes the average of the integral of the charge flux correlation function
        for each molecule type over a specific time period detected from the region
        where the charge flux correlation function is near zero.

        Args:
            tot_flux (array-like): Charge flux correlation function for each molecule type
            integral (array-like): Integral of the charge flux correlation function for
                each molecule type
            tol (float): Tolerance value

        Returns:
            ave (array-like): Average of the integral of the charge flux correlation function
            for each molecule type
            time_range (list): Start and end indexes of the data used for each molecule type
        """
        ave = np.zeros((len(integral)))
        time_range = np.zeros((len(integral)), dtype=object)
        for i in range(len(integral)):
            time_range_ind = self.detect_time_range(tot_flux[i], tol=tol)
            ave[i] = np.average(integral[i][time_range_ind[0] : time_range_ind[1]])
            time_range[i] = (self.time[time_range_ind[0]], self.time[time_range_ind[1]])
        return ave, time_range

    def green_kubo(self, ave):
        """
        Computes the conductivity using the average of the integral of the charge
        correlation function for each molecule type.

        Args:
            ave (array-like): Average of the integral of the charge flux correlation function
            for each molecule type

        Returns:
            cond (array-like): Ionic conductivities
        """
        cond = np.zeros((len(ave)))
        for i in range(len(ave)):
            cond[i] = ave[i] / 3 / constants.BOLTZMANN / self.temp / self.volume
        return cond

    def calc_cond(self, tol=1e-4, plot=False, save=False):
        """
        Wrapper function to calculate the Green-Kubo ionic conductivity from the dumps.

        Args:
            tol (float): Tolerance value to use in detecting the time range over which
                the conductivity is computed
            plot (bool): Whether to plot the data; if True, save a figure with 2 subplots:
                (1) Charge flux correlation function
                (2) Integral of the charge flux correlation function
            save (bool): Whether to save the data as csv files; if True, saves the following:
                (1) Total charge flux correlation as well as the molecular contributions
                as a function of time
                (2) Integral of the total charge flux correlation function as well
                as the molecular contributions as a function of time
                (3) Total and molecular ionic conductivities along with the start and
                end time considered in the calculations

        Returns:
            cond (array-like): Ionic conductivities (S/m) in the same order of molecule
            inputs to PackmolRunner followed by the total conductivity
        """
        j = self.get_charge_flux()
        tot_flux = self.correlate_charge_flux(j)
        integral = self.integrate_charge_flux_correlation(tot_flux)
        ave, time_range = self.fit_curve(tot_flux, integral, tol)
        cond = self.green_kubo(ave)

        if plot:
            time_data = np.array(self.time) * 10 ** 9

            paired = plt.get_cmap("Paired")
            colors = iter(paired(np.linspace(0, 1, 10)))

            fig, ax = plt.subplots(1, 2, figsize=(20, 5), sharey=False)
            ax1 = ax[0]
            set_axis(ax1, axis="both")
            for i in range(len(tot_flux) - 1):
                ax1.plot(
                    time_data, tot_flux[i], linewidth=2, color=next(colors),
                )
            ax1.plot(time_data, tot_flux[-1], linewidth=2, color="black")
            ax1.set_ylabel(r"$\mathrm{\langle J(t)\cdot J(0)\rangle dt}$", fontsize=18)

            paired = plt.get_cmap("Paired")
            colors = iter(paired(np.linspace(0, 1, 10)))
            ax2 = ax[1]
            set_axis(ax2, axis="both")
            for i in range(len(integral) - 1):
                ax2.plot(
                    time_data,
                    integral[i],
                    label=i + 1,
                    linewidth=2,
                    color=next(colors),
                )
            ax2.plot(time_data, integral[-1], label="total", linewidth=2, color="black")
            ax2.legend(
                fontsize=16,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
            )
            ax2.set_ylabel(
                r"$\mathrm{\int_{0}^{\infty}\langle J(t)\cdot J(0)\rangle dt}$",
                fontsize=18,
            )

            for axis in [ax1, ax2]:
                # plots time range used in the total conductivity only
                axis.axvline(
                    time_range[-1][0]*10**9, linewidth=2, color="black", linestyle="--"
                )
                axis.axvline(
                    time_range[-1][1]*10**9, linewidth=2, color="black", linestyle="--"
                )
                axis.set_xscale("log")
                axis.set_xlabel(r"$\mathrm{Time, 10^9 (s)}$", fontsize=18)
                # axis.xaxis.set_major_formatter(ScalarFormatter())
                y_formatter = ScalarFormatter(useOffset=False)
                axis.yaxis.set_major_formatter(y_formatter)
                axis.yaxis.offsetText.set_fontsize(18)
                axis.locator_params(axis="y", nbins=6)

            fig.tight_layout(pad=3)
            fig.savefig(
                f"{self.working_dir}/conductivity.png",
                bbox_inches="tight",
                pad_inches=0.1,
            )
        if save:
            charge_flux = np.append(np.array([self.time]), tot_flux, axis=0)
            integral = np.append(np.array([self.time]), integral, axis=0)
            start_time = [i[0] for i in time_range]
            end_time = [i[1] for i in time_range]
            cond = np.asarray([start_time, end_time, cond])
            mol_names = (
                ",".join([str(i + 1) for i in range(len(tot_flux) - 1)]) + ",tot"
            )
            col_names = "t" + "," + mol_names
            np.savetxt(
                f"{self.working_dir}/charge_flux.csv",
                charge_flux.T,
                delimiter=",",
                header=col_names,
                comments="",
            )
            np.savetxt(
                f"{self.working_dir}/integral.csv",
                integral.T,
                delimiter=",",
                header=col_names,
                comments="",
            )
            np.savetxt(
                f"{self.working_dir}/conductivity.csv",
                cond.T,
                delimiter=",",
                header="start_t,end_t,cond",
                comments="",
            )
        return cond

    def einstein(self):
        pass

    def nernst(self):
        pass
