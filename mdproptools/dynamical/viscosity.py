#!/usr/bin/env python

"""
Calculates viscosity using Green-Kubo relation from LAMMPS output data.
"""
import os
import glob
import random

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize, integrate
from matplotlib.ticker import ScalarFormatter

from pymatgen.io.lammps import outputs

from mdproptools.common import constants

from mdproptools.utilities.plots import set_axis

__author__ = "Matthew Bliss, Rasha Atwi"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "Sep 2021"
__version__ = "0.0.1"


TENSOR_LABELS = ["Pxy", "Pxz", "Pyz"]

# TODO: Implement and test other methods for calculating viscosity (eg. Einstein
#  relation, non-equilibrium methods)


class Viscosity:
    def __init__(
        self,
        log_pattern,
        cutoff_time,
        volume,
        temp=298.15,
        timestep=1,
        acf_method="wkt",
        units="real",
        working_dir=None,
    ):
        self.log_pattern = log_pattern
        self.cutoff_time = cutoff_time
        self.units = units
        self.volume = volume * constants.DISTANCE_CONVERSION[self.units] ** 3
        self.temp = temp
        self.timestep = timestep
        self.acf_method = acf_method
        self.working_dir = working_dir or os.getcwd()
        self.time = None
        self.step_to_s = self.timestep * constants.TIME_CONVERSION[self.units]

    @staticmethod
    def autocorrelate(series, method):
        """
        Returns the time autocorrelation function as defined for using Green-Kubo relations
        :param series: [array-like] The data series to be autocorrelated
        :param method: [str] Method used to calculate autocorrelation function. Options are:
            brute_force: not recommended for large series
            wkt: Wiener-Khinchin theorem as implemented in pylat/src/viscio.py
            Defaults to wht
        :return: [array-like] The autocorrelated series whose indices match those of the input series
        """

        if method == "brute_force":
            normal = np.arange(len(series), 0, -1, dtype="float")
            long_result = np.correlate(series, series, "full")
            result = long_result[long_result.size // 2 :]
            norm_result = np.divide(result, normal)
            acf = norm_result

        elif method == "wkt":
            b = np.concatenate((series, np.zeros(len(series))), axis=0)
            c = np.fft.ifft(np.fft.fft(b) * np.conjugate(np.fft.fft(b))).real
            d = c[: int(len(c) / 2)]
            d = d / (np.array(range(len(series))) + 1)[::-1]
            acf = d

        else:
            raise ValueError("Method string input not recognized")

        return acf

    @staticmethod
    def exp_func(t, A, alpha, tau1, tau2):
        return A * alpha * tau1 * (1 - np.exp(-t / tau1)) + A * (1 - alpha) * tau2 * (
            1 - np.exp(-t / tau2)
        )

    def calc_visc(self, acf, dt):
        integral = integrate.cumtrapz(acf, dx=dt)
        visc = np.multiply(self.volume / (constants.BOLTZMANN * self.temp), integral)
        return visc

    def _calc_3d_visc(self, log_df):
        if self.units not in constants.SUPPORTED_UNITS:
            raise KeyError(
                "Unit type not supported. Supported units are: "
                + str(constants.SUPPORTED_UNITS)
            )

        time_data = log_df["Step"] * self.step_to_s
        delta_t = time_data.iloc[1] - time_data.iloc[0]

        acf_data_list = []
        viscosity_data_list = []

        for label in TENSOR_LABELS:
            pres_1d_data = log_df[label]
            acf_1d_data = (
                self.autocorrelate(pres_1d_data, method=self.acf_method)
                * constants.PRESSURE_CONVERSION[self.units] ** 2
            )
            visc_1d_data = self.calc_visc(acf_1d_data, delta_t)
            acf_data_list.append(acf_1d_data)
            viscosity_data_list.append(visc_1d_data)

        acf_data = np.asarray(acf_data_list)
        viscosity_data = np.asarray(viscosity_data_list)
        viscosity_average = np.mean(viscosity_data, axis=0)
        return viscosity_average, viscosity_data, acf_data

    def calc_avg_visc(self, output_all_data=False):
        list_log_df = []
        log_files = glob.glob(f"{self.working_dir}/{self.log_pattern}")
        for file in log_files:
            log_df = outputs.parse_lammps_log(file)
            list_log_df.append(log_df[0])

        # find the index corresponding to the cutoff_time
        cutoff_time_idx = list_log_df[0].index.get_loc(
            list_log_df[0][list_log_df[0]["Step"] == self.cutoff_time].index[0]
        )

        # calculate viscosity for each replicate
        visc_avg = []
        visc_data = []
        acf_data = []
        for ind, log_df in enumerate(list_log_df):
            print(f"Processing replicate number {ind + 1} out of {len(list_log_df)}")
            log_df = log_df.iloc[cutoff_time_idx:]
            viscosity_average, viscosity_data, acf = self._calc_3d_visc(log_df,)
            visc_avg.append(viscosity_average)
            visc_data.append(viscosity_data)
            acf_data.append(acf)
        self.time = (
            np.array(list_log_df[0]["Step"][: len(visc_avg[0]) - 1]) * self.timestep
        )

        if output_all_data:
            return visc_avg, visc_data, acf_data, self.time
        else:
            return visc_avg

    def fit_avg_visc(
        self,
        visc_avg,
        initial_guess=[1e-10, 0.8, 1.1e4, 1.1e4],
        plot=False,
        plot_file="viscosity.png",
    ):
        visc = np.average(visc_avg, axis=0)
        std = np.std(visc_avg, axis=0)

        time_indexes = np.where(self.time > 2000)
        if time_indexes:
            idx_start_time = time_indexes[0][0]
        else:
            idx_start_time = 1

        std_indexes = np.where(std >= 0.4 * visc)
        if std_indexes:
            idx_cut_time = std_indexes[0][0]
        else:
            idx_cut_time = 1

        popt2, pcov2 = optimize.curve_fit(
            self.exp_func,
            self.time[idx_start_time:idx_cut_time],
            visc[idx_start_time:idx_cut_time],
            sigma=1 / std[idx_start_time:idx_cut_time] ** 0.5,
            bounds=(
                0,
                [
                    max(visc[idx_start_time:idx_cut_time]),
                    1,
                    5 * self.time[idx_cut_time],
                    5 * self.time[idx_cut_time],
                ],
            ),
            p0=initial_guess,
            maxfev=1000000,
        )

        viscosity = (
            popt2[0] * popt2[1] * popt2[2] + popt2[0] * (1 - popt2[1]) * popt2[3]
        )
        fit = []
        for t in self.time[idx_start_time:idx_cut_time]:
            fit.append(self.exp_func(t, *popt2))

        if plot:
            time_data = self.time * self.step_to_s * 10 ** 9

            paired = plt.get_cmap("Paired")
            colors = iter(paired(np.linspace(0, 1, 10)))

            fig, ax = plt.subplots(1, 3, figsize=[20, 5], sharey=False)
            ax1 = ax[0]
            set_axis(ax1, axis="both")
            for visc_arr in visc_avg:
                ax1.plot(
                    time_data, visc_arr[0:-1], linewidth=2, color=next(colors),
                )
            ax1.plot(time_data, visc[0:-1], linewidth=2, color="black")
            ax1.axvline(
                time_data[idx_cut_time], linewidth=2, color="black", linestyle="--"
            )
            ax1.set_xlabel(r"$\mathrm{Time, 10^9 (m^2/s)}$", fontsize=18)
            ax1.set_ylabel(r"$\mathrm{\mu \ (Pa.s)}$", fontsize=18)

            ax2 = ax[1]
            set_axis(ax2, axis="both")
            ax2.plot(time_data, std[0:-1], linewidth=2, color="black")
            ax2.set_xlabel(r"$\mathrm{Time, 10^9 (s)}$", fontsize=18)
            ax2.set_ylabel(r"$\mathrm{\sigma \ (Pa.s)}$", fontsize=18)

            ax3 = ax[2]
            set_axis(ax3, axis="both")
            ax3.plot(
                time_data[idx_start_time:idx_cut_time],
                visc[idx_start_time:idx_cut_time],
                linewidth=2,
                color="red",
                label="data",
            )
            ax3.plot(
                time_data[idx_start_time:idx_cut_time],
                fit,
                linewidth=2,
                color="black",
                label="fit",
            )
            # TODO: add fit parameters
            # ax3.annotate(
            #     r"$\mathrm{\mu = }$" + str(round(viscosity, 4)) + "Pa.s",
            #     xy=(0.5, 0),
            #     fontsize=16,
            #     textcoords='offset points',
            #     xycoords=('axes fraction', 'figure fraction'),
            #     xytext=(0, 10),
            #     ha="center",
            #     va="bottom",
            # )
            ax3.legend(fontsize=16, loc="lower right", frameon=False)
            ax3.set_xlabel(r"$\mathrm{Time, 10^9 (m^2/s)}$", fontsize=18)
            ax3.set_ylabel(r"$\mathrm{\mu \ (Pa.s)}$", fontsize=18)

            for axis in [ax1, ax2, ax3]:
                axis.xaxis.set_major_formatter(ScalarFormatter())
                y_formatter = ScalarFormatter(useOffset=False)
                axis.yaxis.set_major_formatter(y_formatter)
                axis.yaxis.offsetText.set_fontsize(18)
                axis.locator_params(axis="y", nbins=6)

            fig.tight_layout(pad=3)
            fig.savefig(
                f"{self.working_dir}/{plot_file}", bbox_inches="tight", pad_inches=0.1
            )

        return viscosity

    def bootstrapping(
        self,
        visc_avg,
        num_replicates,
        tot_replicates,
        initial_guess=[1e-10, 0.8, 1.1e4, 1.1e4],
        plot=True,
    ):

        idx = np.zeros((tot_replicates, num_replicates), dtype=int)
        for i in range(tot_replicates):
            idx[i] = random.sample(range(len(visc_avg)), num_replicates)
        visc_samples = np.array(visc_avg)[idx]

        all_visc = []
        for ind, visc in enumerate(visc_samples):
            print(f"Fitting viscosity sample {ind + 1} out of {len(visc_samples)}")
            viscosity = self.fit_avg_visc(
                visc_avg=visc,
                initial_guess=initial_guess,
                plot=plot,
                plot_file=f"viscosity_{ind + 1}.png",
            )
            all_visc.append(viscosity)
        final_viscosity = np.average(all_visc)
        final_std = np.std(all_visc)
        return final_viscosity, final_std
