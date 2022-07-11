#!/usr/bin/env python

"""
Calculates the residence time from LAMMPS trajectory files.
"""

import os
import time

import numba as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gamma
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acovf

from pymatgen.io.lammps.outputs import parse_lammps_dumps

from mdproptools.utilities.plots import set_axis
from mdproptools.structural.rdf_cn import _calc_rsq

__author__ = "Rasha Atwi"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "May 2022"
__version__ = "0.0.2"


@nb.njit(cache=True)
def find_intersection(a, b):
    res = 0
    for i in a:
        res += b[i[0], i[1]]
    return res


# TODO: COM - sanity checks (wrapped coords ...) - unique atom ids - unit conversion
class ResidenceTime:
    def __init__(self, r_cut, partial_relations, filename, dt=1, working_dir=None):
        self.r_cut = r_cut
        self.relation_matrix = np.asarray(partial_relations).transpose()
        self.atom_pairs = []
        self.dumps = parse_lammps_dumps(filename)
        self.dt = dt * 10 ** -3  # input dt in fs - convert to ps
        self.corr_df = None
        self.res_time_df = None
        self.working_dir = working_dir or os.getcwd()

    @staticmethod
    def _stretched_exp_function(x, tau, beta):
        return np.exp(1 - (1 + x / tau) ** beta)

    @staticmethod
    def _integrate_sum_exp(tau, beta):
        return np.exp(1) * tau * gamma(1 + 1 / beta)

    def calc_auto_correlation(self):
        num_of_atom_pair_atoms = {}
        id_list = None
        h_matrix_dict = {}
        correlation = {"Time (ps)": []}

        start = time.time()
        for ind, dump in enumerate(self.dumps):
            print("Processing frame number: {}".format(ind))
            # dump.data = dump.data.sort_values(by=["id"])
            correlation["Time (ps)"].append(dump.timestep * self.dt)

            lx, ly, lz = dump.box.to_lattice().lengths
            # full_df = dump.data[["type", "x", "y", "z"]]
            full_df = dump.data[["id", "type", "x", "y", "z"]]
            if ind == 0:
                id_list = full_df['id'].to_list()
            full_df = full_df.set_index('id').reindex(id_list)
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
                    h_index_of_true_values = np.nonzero(h)
                    h_matrix.append(list(h_index_of_true_values[0]))
                h_matrix_dict[atom_pair] = h_matrix_dict.get(atom_pair, []) + [h_matrix]
        end = time.time()
        print(f"First loop took: {end - start}")

        cl = int(len(correlation["Time (ps)"]) / 2)
        correlation["Time (ps)"] = correlation["Time (ps)"]

        start = time.time()
        for kl in range(0, len(self.relation_matrix)):
            k, l = self.relation_matrix[kl]
            atom_pair = f"{k}-{l}"

            h_matrix = h_matrix_dict.pop(atom_pair)
            number_of_time_steps = len(h_matrix)
            number_of_central_atoms = len(h_matrix[0])
            total_number_of_columns = 0
            cov_mat = []
            for central_atom in range(number_of_central_atoms):
                central_atom_h_matrix = [[i for i in j[central_atom]] for j in h_matrix]
                max_cols = [max(i) for i in central_atom_h_matrix if i]
                if not max_cols:
                    continue
                np_h_matrix = np.zeros(
                    (
                        len(h_matrix),
                        max([max(i) for i in central_atom_h_matrix if i]) + 1,
                    ),
                    dtype="bool",
                )
                for row in range(number_of_time_steps):
                    np_h_matrix[row, list(h_matrix[row][central_atom])] = True
                total_number_of_columns += np_h_matrix.shape[1]
                cov_array = np.zeros((number_of_time_steps,))

                for column in range(np_h_matrix.shape[1]):
                    cov_array += acovf(
                        np_h_matrix[:, column], demean=False, unbiased=True, fft=True
                    )
                cov_mat.append(cov_array)
                del np_h_matrix
            corr_array = np.sum(np.array(cov_mat), axis=0) / total_number_of_columns
            del cov_mat
            corr_array = corr_array / corr_array[0]
            correlation[atom_pair] = corr_array
        end = time.time()
        print(f"Second loop took: {end - start}")

        self.corr_df = pd.DataFrame.from_dict(correlation)
        self.corr_df.to_csv(self.working_dir + "/auto_correlation.csv")

    def fit_auto_correlation(self, cut_percent=0.5, plot=True):
        residence_time = {}
        corr_data = self.corr_df.head(
            int(len(self.corr_df) * cut_percent)
        )  # take first half of the data
        for col in corr_data:
            if col != "Time (ps)":
                x = corr_data["Time (ps)"].values
                y = corr_data[col].values

                popt, _ = curve_fit(self._stretched_exp_function, x, y,
                                    bounds=([0, 0.1], [np.inf, 1]))
                tau, beta = popt

                residence_time[col] = [
                    tau,
                    beta,
                    self._integrate_sum_exp(tau, beta),
                ]
                if plot:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    set_axis(ax)
                    ax.scatter(
                        corr_data["Time (ps)"],
                        corr_data[col],
                        color="red",
                        label="original",
                    )
                    fit_data = self._stretched_exp_function(
                        corr_data["Time (ps)"].values, tau, beta
                    )
                    ax.plot(
                        corr_data["Time (ps)"], fit_data, color="black", label="fit"
                    )
                    ax.legend(frameon=False, fontsize=20)
                    ax.set_xlabel("Time (ps)", fontsize=20)
                    ax.set_ylabel("C(t)", fontsize=20)
                    fig.savefig(
                        self.working_dir + f"/{col}_fit.png",
                        bbox_inches="tight",
                        pad_inches=0.1,
                    )
                    plt.close()
        print("Finished computing residence time")
        self.res_time_df = pd.DataFrame(residence_time)
        self.res_time_df.index = ["tau", "beta", "r (ps)"]
        self.res_time_df.to_csv(self.working_dir + "/residence_time.csv")
        return residence_time


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
