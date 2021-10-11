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

from pymatgen.io.lammps.outputs import parse_lammps_log

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
    """
    Class to calculate the viscosity of a solution from MD simulations. Uses
    Green-Kubo correlation and follows the methods in: 10.1021/acs.jcim.9b00066 and
    10.1021/acs.jctc.5b00351. Supports calculating the viscosity from one trajectory
    or multiple replicates to get a statistical average and std. If multiple replicates
    are available, bootstrapping can be done to obtain a distribution of viscosity
    values. Fits the running integral viscosity to a double exponential function which
    it analytically integrates to extrapolate to
    infinite time.
    """
    
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
        """
        Creates a Viscosity object. 

        Args:
            log_pattern (str): Pattern of the name of the LAMMPS log files
            cutoff_time (int): Simulation time to ignore in the autocorrelation
            volume (float): Volume of the simulation box in the same units specified as input
            temp (flaot): Temperature (K); defaults to 298.15 K
            timestep (int or float): Timestep used in the simulations in the same units
                specified as input; defaults to 1 fs when real units are used
            acf_method (str): Method used to calculate the autocorrelation function.
                Options are:
                brute_force: Not recommended for large series
                wkt: Wiener-Khinchin theorem as implemented in pylat/src/viscio.py
                Defaults to wkt
            units (str): Units used in the LAMMMPS simulations; used to convert to SI
                units; defaults to real unit
            working_dir (str): Path of the LAMMPS log files
        """
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
        Returns the time autocorrelation function as defined for using Green-Kubo relations.

        Args:
            series (array-like): Data series to be correlated
            method (str): Method used to calculate the autocorrelation function.
                Options are:
                brute_force: Not recommended for large series
                wkt: Wiener-Khinchin theorem as implemented in pylat/src/viscio.py
                Defaults to wkt

        Returns:
            Array of the autocorrelated series whoses indices match thoses of the
            input series
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
        """
        Defines a double exponential function used in fitting the viscosity data.
        Based on 10.1021/jp062885s.

        Args:
            t (int or float): Time
            A, alpa, tau1, tau2 (float): Fitting parameters

        Returns:
            Value of the double exponential function
        """
        return A * alpha * tau1 * (1 - np.exp(-t / tau1)) + A * (1 - alpha) * tau2 * (
            1 - np.exp(-t / tau2)
        )

    def calc_visc(self, acf, dt):
        """
        Calculates the viscosity by integrating the 1-D pressure tensor autocorrelation
        function over time.

        Args:
            acf (array-like): Pressure tensor autocorrelation data
            dt (array-like): Time data

        Returns:
            Array of 1-D viscosity data
        """
        integral = integrate.cumtrapz(acf, dx=dt)
        visc = np.multiply(self.volume / (constants.BOLTZMANN * self.temp), integral)
        return visc

    def _calc_3d_visc(self, log_df):
        """
        Calculates the viscosity using all the elements of the pressure tensor.

        Args:
            log_df (pd.Datafarme): Thermo dataframe from the LAMMPS log file

        Returns:
            Arrays of the average viscosity, viscosity from each element of the
            pressure tensore, and autocorrelation data as a function of time
        """
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
        """
        Parses LAMMPS log files (replicates) for the pressure tensors and calculates the
        viscosity from each replicate while ignoring the first few steps corresponding
        to the cutoff_time.

        Args:
            output_all_data (bool): Whether to output the average viscosity, viscosity
                from each pressure tensor, autocorrelation function, and time data or
                just output the average viscosity; defaults to False

        Returns:
            Arrays of average viscosity per replicate and/or viscosity from each
            pressure tensor, autocorrelation data, and time
        """
        list_log_df = []
        log_files = glob.glob(f"{self.working_dir}/{self.log_pattern}")
        for file in log_files:
            log_df = parse_lammps_log(file)
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
        """
        Computes the average and std of the running time integrals corresponding to
        different replicates; fits the average viscosity to a double exponential function
        using least-squares regression with a weighting function 1/std**0.5; then
        computes the final value for the viscosity from the infinite-time value of the
        double exponential function; ignores the first 2 ps in the fitting due to large
        fluctuations and uses a long-time cutoff corresponding to a std <= 0.4*running
        average viscosity. Works for one or multiple replicates.

        Args:
            visc_avg (array-like): Average viscosity data as a function of time for
                different replicates
            initial_guess (list): Initial guess for the double exponential function
                parameters
            plot (bool): Whether to plot the viscosity data; if True, saves a figure
                with the 3 subplots:
                (1) Viscosity data versus time from different replicates along with the
                average running integral and a vertical line corresponding to the
                cutoff time used
                (2) Std versus time
                (3) Viscosity data and viscosity fit versus time
                Defaults to False
            plot_file (str): Name of the plot file

        Returns:
            Infinite-time value of the viscosity
        """
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
            colors = iter(paired(np.linspace(0, 1, 50)))

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
            ax1.set_ylabel(r"$\mathrm{\mu \ (Pa.s)}$", fontsize=18)

            ax2 = ax[1]
            set_axis(ax2, axis="both")
            ax2.plot(time_data, std[0:-1], linewidth=2, color="black")
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
            # TODO: add fit parameters to plot
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
            ax3.set_ylabel(r"$\mathrm{\mu \ (Pa.s)}$", fontsize=18)

            for axis in [ax1, ax2, ax3]:
                axis.set_xlabel(r"$\mathrm{Time, 10^9 (m^2/s)}$", fontsize=18)
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
        """
        Performs bootstrapping by using random sampling of the running integrals from
        different replicates to obtain a distribution of viscosity values from which it
        calculates an estimate of the uncertainty; does not use the same replicate
        multiple times within one iteration

        Args:
            visc_avg (array-like): Average viscosity data as a function of time for
                different replicates
            num_replicates (int): Number of replicates to randomly choose for each
                iteration of the bootstrapping method
            tot_replicates (int): Number of bootstrapping iterations to run
            initial_guess (list): Initial guess for the double exponential function
                parameters
            plot (bool): Whether to plot the viscosity data; if True, saves a figure
                with the 3 subplots:
                (1) Viscosity data versus time from different replicates along with the
                average running integral and a vertical line corresponding to the
                cutoff time used
                (2) Std versus time
                (3) Viscosity data and viscosity fit versus time
                Defaults to False

        Returns:
            Final viscosity and std estimates
        """

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
