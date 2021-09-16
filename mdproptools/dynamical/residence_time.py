#!/usr/bin/env python

"""
Calculates the residence time from LAMMPS trajectory files.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from pymatgen.io.lammps.outputs import parse_lammps_dumps

from mdproptools.utilities.plots import set_axis
from mdproptools.structural.rdf_cn import _calc_rsq

__author__ = "Rasha Atwi"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "Mar 2021"
__version__ = "0.0.1"


# TODO: COM - sanity checks (wrapped coords ...) - unique atom ids -
#  begin and end time (ignore first ps in fitting) - multi-exponential fitting
class ResidenceTime:
    def __init__(self, r_cut, partial_relations, filename, dt=1, working_dir=None):
        self.r_cut = r_cut
        self.relation_matrix = np.asarray(partial_relations).transpose()
        self.atom_pairs = []
        self.dumps = list(parse_lammps_dumps(filename))
        self.dt = dt * 10 ** -3  # input dt in fs - convert to ps
        self.corr_df = None
        self.res_time_df = None
        self.working_dir = working_dir or os.getcwd()

    @staticmethod
    def _exp_func(t, a1, a2, a3, a4, tau1, tau2, tau3, tau4):
        return (
            a1 * np.exp(-t / tau1)
            + a2 * np.exp(-t / tau2)
            + a3 * np.exp(-t / tau3)
            + a4 * np.exp(-t / tau4)
        )

    def calc_auto_correlation(self):
        h_init = {}
        sum_h_init = {}
        h_matrix_dict = {}
        cl = int(len(self.dumps) / 2)
        correlation = {"Time (ps)": []}
        for ind, dump in enumerate(self.dumps):
            print("Processing frame number: {}".format(ind))
            if ind < cl:
                correlation["Time (ps)"].append(dump.timestep * self.dt)
            lx, ly, lz = dump.box.to_lattice().lengths
            full_df = dump.data[["type", "x", "y", "z"]]
            for kl in range(0, len(self.relation_matrix)):
                k, l = self.relation_matrix[kl]
                atom_pair = f"{k}-{l}"
                self.atom_pairs.append(atom_pair)
                k_data = full_df[full_df["type"] == k].drop("type", axis=1).values
                l_data = full_df[full_df["type"] == l].drop("type", axis=1).values
                h_matrix = []
                for idx, k_row in enumerate(k_data):
                    data_i, rsq = _calc_rsq(k_row, l_data, lx, ly, lz, 0)
                    h = (rsq > self.r_cut[kl][0] ** 2) & (rsq <= self.r_cut[kl][1] ** 2)
                    if k == l:
                        h[idx] = False
                    h_matrix.append(h.reshape(1, len(h)))
                h_matrix = np.concatenate(h_matrix, axis=0)
                h_matrix_dict[atom_pair] = h_matrix_dict.get(atom_pair, []) + [h_matrix]
                if atom_pair not in h_init:
                    h_init[atom_pair] = h_matrix.copy()
                    sum_h_init[atom_pair] = np.sum(h_init[atom_pair] ** 2)
                    if sum_h_init[atom_pair] == 0:
                        sum_h_init[atom_pair] = 0.1
            print(
                "Finished computing auto-correlation function for timestep",
                dump.timestep,
            )
        for kl in range(0, len(self.relation_matrix)):
            k, l = self.relation_matrix[kl]
            atom_pair = f"{k}-{l}"
            correlation[atom_pair] = [0] * cl
            for delta_time in range(cl):
                for i in range(cl):
                    s = (
                        np.sum(
                            h_matrix_dict[atom_pair][i]
                            * h_matrix_dict[atom_pair][i + delta_time]
                        )
                        / sum_h_init[atom_pair]
                    )
                    correlation[atom_pair][delta_time] += s
                correlation[atom_pair][delta_time] /= cl
        self.corr_df = pd.DataFrame.from_dict(correlation)
        self.corr_df.to_csv(self.working_dir + "/auto_correlation.csv")

    def fit_auto_correlation(self):
        residence_time = {}
        a_opt = [f"a{i}" for i in range(1, 5)]
        tau_opt = [f"tau{i}" for i in range(1, 5)]
        for col in self.corr_df:
            if col != "Time (ps)":
                residence_time[col] = {}
                x = self.corr_df["Time (ps)"]
                y = self.corr_df[col]
                popt, _ = curve_fit(self._exp_func, x, y, maxfev=1000000)
                fit = self._exp_func(x, *popt)
                for i, j in enumerate(a_opt + tau_opt):
                    residence_time[col][j] = popt[i]
                residence_time[col]["error"] = np.sum((y - fit) ** 2)
                residence_time[col]["residence_time (ps)"] = np.sum(
                    np.multiply(popt[0:4], popt[4:])
                )
        print("Finished computing residence time")
        self.res_time_df = pd.DataFrame(residence_time)
        self.res_time_df.to_csv(self.working_dir + "/residence_time.csv")

    def plot_results(self):
        for atom_pair in self.atom_pairs:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                set_axis(ax)
                fit = self._exp_func(
                    self.corr_df["Time (ps)"], *self.res_time_df[atom_pair][0:8].values
                )
                ax.scatter(
                    self.corr_df["Time (ps)"],
                    self.corr_df[atom_pair],
                    color="red",
                    label="original",
                )
                ax.plot(self.corr_df["Time (ps)"], fit, color="black", label="fit")
                ax.legend(frameon=False, fontsize=20)
                ax.set_xlabel("Time (ps)", fontsize=20)
                ax.set_ylabel("C(t)", fontsize=20)
                fig.savefig(
                    self.working_dir + f"/{atom_pair}_fit.png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
            except Exception as e:
                print(e)


class Displacement:
    def __init__(
        self,
        atom_types,
        residence_time,
        filename,
        dt=1,
        save_mode=True,
        working_dir=None,
    ):
        self.atom_types = atom_types
        self.residence_time = residence_time
        self.dumps = list(parse_lammps_dumps(filename))
        self.dt = dt * 10 ** -3  # input dt in fs
        self.save_mode = save_mode
        self.working_dir = working_dir or os.getcwd()

    def calc_dist(self):
        atoms_data = {}
        for ind, dump in enumerate(self.dumps):
            full_df = dump.data[["id", "type", "x", "y", "z"]]
            full_df = full_df.sort_values("id")
            for atom_type in self.atom_types:
                atoms_coord = full_df[full_df["type"] == atom_type][
                    ["id", "x", "y", "z"]
                ]
                atoms_coord["Time (ps)"] = dump.timestep * self.dt
                atoms_data[atom_type] = atoms_data.get(atom_type, []) + [atoms_coord]
        for time, (key, value) in zip(atoms_data.items()):
            atoms_data[key] = pd.concat(value)

        print(atoms_data)
        # atoms_coord = \
        #     full_df[full_df['type'] == atom_type][['x', 'y', 'z']].values
        # atoms_data[atom_type] = \
        #     atoms_data.get(atom_type, []) + [atoms_coord]

        # for key, value in atoms_data.items():
        #     if key != 'Time (ps)':
        #         df = pd.DataFrame.from_dict()
        #         df.groupby(pd.Grouper(key='Time (ps)',
        #                               freq=self.residence_time[key]))

        # return atoms_data
