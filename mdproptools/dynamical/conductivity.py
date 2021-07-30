import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import cumtrapz
from matplotlib.ticker import ScalarFormatter

from pymatgen.io.lammps.outputs import parse_lammps_dumps

from mdproptools.dynamical.residence_time import _set_axis


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

    def define_mol_cols(self, dump):
        # TODO: this is exactly the same as the one in diffusivity except for
        #  vx instead of xu; need to merge and put in a common module
        # TODO: speedup this - it is the most exp part of cond calc
        mol_types = []
        mol_ids = []

        for mol_type, number_of_mols in enumerate(self.num_mols):
            for mol_id in range(number_of_mols):
                for atom_id in range(self.num_atoms_per_mol[mol_type]):
                    mol_types.append(mol_type + 1)
                    mol_ids.append(mol_id + 1)
        if not self.mass:
            assert "mass" in dump.data.columns, "Missing atom masses in dump file."
            df = pd.DataFrame(
                dump.data[["id", "type", "vx", "vy", "vz", "mass", "q"]],
                columns=["id", "type", "vx", "vy", "vz", "mass", "q"],
            )
        else:
            df = pd.DataFrame(
                dump.data[["id", "type", "vx", "vy", "vz", "q"]],
                columns=["id", "type", "vx", "vy", "vz", "q"],
            )
            df["mass"] = df.apply(lambda x: self.mass[int(x.type - 1)], axis=1)
        df["mol_type"] = mol_types
        df["mol_id"] = mol_ids
        df = df.drop(["type", "id"], axis=1).set_index(["mol_type", "mol_id"])
        mol_df = df.groupby(["mol_type", "mol_id"]).apply(
            lambda x: pd.Series(
                x["mass"].values @ x[["vx", "vy", "vz"]].values / x.mass.sum(),
                index=["vx", "vy", "vz"],
            )
        )
        mol_df["mass"] = df.groupby(["mol_type", "mol_id"])["mass"].sum()
        mol_df["q"] = df.groupby(["mol_type", "mol_id"])["q"].sum()
        return mol_df.reset_index().rename(columns={"mol_type": "type"})

    def get_charge_flux(self):
        for ind, dump in enumerate(self.dumps):
            print("Processing frame number: {}".format(ind))
            # if ind < self.cl:
            #     self.time.append(dump.timestep * self.dt)
            self.time.append(dump.timestep)
            dump.data = dump.data.sort_values(by=["id"])
            dump.data.reset_index(inplace=True)
            # get dataframe with mol type, mol id, vx, vy, vz, mass, q
            df = self.define_mol_cols(dump)

            def dot(x):
                return np.dot(x, df.loc[x.index, "q"])

            # for each mol type at a given time, calculate charge flux
            flux = df.groupby("type").agg({"vx": dot, "vy": dot, "vz": dot})
            self.j[:, :, ind] = flux.reset_index().drop("type", axis=1).T.values
            print("Finished computing charge flux for timestep", dump.timestep)

    # def _correlate_charge_flux(self):
    #     # TODO: np.correlate, put in common module (similar method in r.t)
    #     # calculate contribution from each molecule type
    #     self.tot_flux = np.zeros((len(self.num_mols) + 1, self.cl))
    #     for i in range(len(self.num_mols)):
    #         for j in range(len(self.num_mols)):
    #             for delta_t in range(self.cl):
    #                 for t in range(self.cl):
    #                     self.tot_flux[i, delta_t] += \
    #                         np.sum(self.j[:, i, t] * self.j[:, j, t + delta_t])
    #                 self.tot_flux[i, delta_t] /= self.cl
    #     # add the total flux to the end
    #     self.tot_flux[-1, :] = self.tot_flux[0:-1, :].sum(axis=0)

    def correlate_charge_flux(self):
        # TODO: np.correlate, put in common module (similar method in r.t)
        # calculate contribution from each molecule type
        for i in range(len(self.num_mols)):
            for j in range(len(self.num_mols)):
                for k in range(self.j.shape[0]):
                    corr = self._correlate(self.j[k, i, :], self.j[k, j, :])
                    self.tot_flux[i, :] += corr
                    self.tot_flux[-1, :] += corr

    @staticmethod
    def _correlate(a, b):
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
        # TODO: provide time range for each type (tot and mols)
        for i in range(len(self.integral)):
            if not time_range:
                self.ave[i] = np.average(self.integral[i])
            else:
                begin_ind = np.where(np.array(self.time) == time_range[0])[0][0]
                end_ind = np.where(np.array(self.time) == time_range[1])[0][0]
                self.ave[i] = np.average(self.integral[i][begin_ind:end_ind])

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
