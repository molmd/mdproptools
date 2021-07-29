import numpy as np
import analysis.lammps.constants as cnst
from scipy import integrate


__author__ = 'Matthew Bliss'
__version__ = '0.1'
__email__ = 'Matthew.Bliss@tufts.edu'
__date__ = 'Jun 10, 2020'


TENSOR_LABELS = ['Pxy', 'Pxz', 'Pyz']


def autocorrelate(series,method = 'wht'):
    '''
    Returns the time autocorrelation function as defined for using Green-Kubo relations
    :param series: [array-like] The data series to be autocorrelated
    :param method: [str] Method used to calculate autocorrelation function. Options are:
        brute_force: not recommended for large series
        wkt: Wiener-Khinchin theorem as implemented in pylat/src/viscio.py
        Defaults to wht
    :return: [array-like] The autocorrelated series whose indices match those of the input series
    '''

    if method == 'brute_force':
        normal = np.arange(len(series),0,-1,dtype='float')
        long_result = np.correlate(series,series,'full')
        result = long_result[long_result.size // 2:]
        norm_result = np.divide(result,normal)
        acf = norm_result

    elif method == 'wkt':
        b = np.concatenate((series, np.zeros(len(series))), axis=0)
        c = np.fft.ifft(np.fft.fft(b) * np.conjugate(np.fft.fft(b))).real
        d = c[:int(len(c) / 2)]
        d = d / (np.array(range(len(series))) + 1)[::-1]
        acf = d

    else:
        raise ValueError('Method string input not recognized')

    return acf

def calc_visc(acf, volume, temperature, dt):
    ''''''
    integral = integrate.cumtrapz(acf,dx=dt)
    visc = np.multiply(volume / (cnst.BOLTZMANN * temperature), integral)
    return visc

def calc_3d_visc(log_df, volume, temp, timestep, acf_method='wkt', units='real', output_all_data=False):
    ''''''
    if units not in cnst.SUPPORTED_UNITS:
        raise KeyError('Unit type not supported. Supported units are: ' + str(cnst.SUPPORTED_UNITS))

    step_to_s = timestep * cnst.TIME_CONVERSION[units]
    time_data = log_df['Step'] * step_to_s
    delta_t = time_data.iloc[1] - time_data.iloc[0]

    si_volume = volume * cnst.DISTANCE_CONVERSION[units]**3

    acf_data_list = []
    viscosity_data_list = []

    for label in TENSOR_LABELS:
        pres_1d_data = log_df[label]
        acf_1d_data = autocorrelate(pres_1d_data, method=acf_method) * cnst.PRESSURE_CONVERSION[units]**2
        visc_1d_data = calc_visc(acf_1d_data, si_volume, temp, delta_t)
        acf_data_list.append(acf_1d_data)
        viscosity_data_list.append(visc_1d_data)

    acf_data = np.asarray(acf_data_list)
    viscosity_data = np.asarray(viscosity_data_list)

    viscosity_average = np.mean(viscosity_data, axis=0)

    if output_all_data:
        return viscosity_average, viscosity_data, acf_data
    else:
        return viscosity_average

#TODO: Implement and test other methods for calculating viscosity (eg. Einstein relation, non-equilibrium methods)