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

from mdproptools.common.com_mols import calc_com
from mdproptools.dynamical.residence_time import _set_axis

__author__ = "Rasha Atwi"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "Jan 2021"
__version__ = "0.0.1"


class Conductivity:
    def __init__(
        self,
        filename,
        num_mols,
        num_atoms_per_mol,
        mass=None,
        timestep=1,
        working_dir=None,
    ):
        self.working_dir = working_dir or os.getcwd()
        self.dumps = list(parse_lammps_dumps(f"{self.working_dir}/{filename}"))
        self.dt = timestep
        self.mass = mass
        self.num_mols = num_mols
        self.num_atoms_per_mol = num_atoms_per_mol
        box_lengths = self.dumps[0].box.to_lattice().lengths
        self.volume = np.prod(box_lengths) / 10 ** 30  # volume in m^3
        # prepare empty charge flux of shape (xyz, # molecule types, # steps)
        self.j = np.zeros((3, len(self.num_mols), len(self.dumps)))
        self.tot_flux = np.zeros((len(self.num_mols) + 1, self.j.shape[2]))
        self.integral = np.zeros((len(self.tot_flux), len(self.tot_flux[0])))
        self.ave = np.zeros((len(self.integral)))
        self.cond = np.zeros((len(self.ave)))
        # number of points for averaging correlations over multiple time origins
        self.cl = int(len(self.dumps) / 2)
        self.time = []  # time data used to calculate GK integral

    def get_charge_flux(self):
        inputs = []
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
            self.time.append(i[0])
            self.j[:, :, i[1]] = i[2]

    def correlate_charge_flux(self):
        # calculate contribution from each molecule type
        for i in range(len(self.num_mols)):
            for j in range(len(self.num_mols)):
                for k in range(self.j.shape[0]):
                    corr = self.correlate(self.j[k, i, :], self.j[k, j, :])
                    self.tot_flux[i, :] += corr
                    self.tot_flux[-1, :] += corr

    @staticmethod
    def correlate(a, b):
        al = np.concatenate((a, np.zeros(len(a))), axis=0)
        bl = np.concatenate((b, np.zeros(len(b))), axis=0)
        c = np.fft.ifft(np.fft.fft(al) * np.conjugate(np.fft.fft(bl))).real
        d = c[: len(c) // 2]
        d /= (np.arange(len(d)) + 1)[::-1]
        return d

    def integrate_charge_flux_correlation(self):
        delta = self.dt * (self.time[1] - self.time[0])
        for i in range(0, len(self.tot_flux)):
            self.integral[i][1:] = cumtrapz(self.tot_flux[i], dx=delta)

    def fit_curve(self, time_range=None):
        for i in range(len(self.integral)):
            time_range_ind = self.detect_time_range(self.integral[i], tol=tol)
            self.ave[i] = np.average(self.integral[i][time_range_ind[0]:time_range_ind[1]])

    def green_kubo(self, temp=298.15):
        k = 1.38e-23
        el = 1.60217e-19
        for i in range(len(self.ave)):
            self.cond[i] = self.ave[i] / 3 / k / temp / self.volume * el ** 2 / 10 ** 5

    def save(self):
        charge_flux = np.append(np.array([self.time]), self.tot_flux, axis=0)
        integral = np.append(np.array([self.time]), self.integral, axis=0)
        mol_names = ",".join([str(i + 1) for i in range(len(self.tot_flux))])
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
            self.cond.reshape(1, self.cond.shape[0]),
            delimiter=",",
            header=mol_names,
            comments="",
        )

    def plot(self):
        paired = plt.get_cmap("Paired")
        colors = iter(paired(np.linspace(0, 1, 10)))
        fig, ax = plt.subplots(figsize=(8, 6))
        _set_axis(ax, axis="both")
        for i in range(len(self.tot_flux)):
            ax.plot(
                np.array(self.time),
                self.tot_flux[i],
                label=i,
                linewidth=2,
                color=next(colors),
            )
        ax.legend(fontsize=18, frameon=False)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        y_formatter = ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(y_formatter)
        ax.yaxis.offsetText.set_fontsize(18)
        ax.set_xlabel("Time (ps)", fontsize=18)
        ax.set_ylabel(r"$\mathrm{\langle J(t)\cdot J(0)\rangle dt}$", fontsize=18)
        fig.tight_layout()
        fig.savefig(
            f"{self.working_dir}/tot_flux.png", bbox_inches="tight", pad_inches=0.1
        )

        paired = plt.get_cmap("Paired")
        colors = iter(paired(np.linspace(0, 1, 10)))
        fig, ax = plt.subplots(figsize=(8, 6))
        _set_axis(ax, axis="both")
        for i in range(len(self.integral)):
            ax.plot(
                np.array(self.time),
                self.integral[i],
                label=i,
                linewidth=2,
                color=next(colors),
            )
        ax.legend(fontsize=18, frameon=False)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        y_formatter = ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(y_formatter)
        ax.yaxis.offsetText.set_fontsize(18)
        ax.set_xlabel("Time (ps)", fontsize=18)
        ax.set_ylabel(
            r"$\mathrm{\int_{0}^{\infty}\langle J(t)\cdot J(0)\rangle dt}$", fontsize=18
        )
        fig.tight_layout()
        fig.savefig(
            f"{self.working_dir}/integral.png", bbox_inches="tight", pad_inches=0.1
        )

    def einstein(self):
        pass

    def nernst(self):
        pass
