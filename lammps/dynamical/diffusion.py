import re
import numpy as np
from monty.io import zopen
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from pymatgen.io.lammps.outputs import parse_lammps_dumps

__author__ = 'Rasha Atwi, Matthew Bliss'
__version__ = '0.2'
__email__ = 'rasha.atwi@tufts.edu, Matthew.Bliss@tufts.edu'
__date__ = 'Aug 8, 2019'


def get_msd(filename):
    """
    This function reads a text file containing msd data and saves it to a dictionary.
    :param filename: name of the file containing time (fs) and msd data (Angstrom^2)
    :return: dictionary with keys of time (s) and values of msd data (cm^2)
    """
    #msd_patt = re.compile(r"(Step)\s*(\w*\s*)(\wmsd)\D")
    #tab_patt = re.compile(r"^\s*(\d+)\s*(\d+.\d*)\s*(\d+.\d*)\s*(\d+.\d*)\D")

    msd = {}

    read_msd = False
    number_of_cols = None
    idx = None
    with zopen(filename) as f:
        for line in f:
            if line.startswith('Step') and 'msd' in line:
                read_msd = True
                line_split = line.split()
                number_of_cols = len(line_split)
                idx = [i for i, j in enumerate(line_split) if 'msd' in j]
                continue
            if read_msd:
                line_split = line.split()
                if len(line_split) != number_of_cols or re.search('(?![eE])[a-zA-Z]', line): #exclude e from string search
                    break
                # Converting time from fs to s and msd from Angstrom^2 to cm^2
                time = int(line_split[0]) / 10 ** 15
                disp = [float(line_split[i]) / 10 ** 16 for i in idx]
                msd[time] = disp
    return msd

def get_msd_from_dump(file_pattern,fs_per_step=1,msd_type='com',one_d=False,mol_species='all',initial_timestep=0,output=False):
    '''
    This function reads dump files (single or multiple frames per file), calculates msd, and saves it to a dictionary. Assumes real units.
    Does not recognize scaled positions (sx, sxu, etc.). Currently only supports msd based on all atoms in system or based on CoM of all molecules in system.
    Will want to update in future for msd based on single molecular species.

    :param file_pattern (str): If using dump file w/ multiple frames per file, this is the filename.
        If using dump files w/ single frame per file, this should be a pattern that glob can read and sort such that the earliest frame should be first.

    :param fs_per_step (float/int): The number of fs per frame; same as 'Timestep' command in control file. Defaults to 1.

    :param msd_type (str): Controls the method used to calculate msd. If 'com', calculates based on center of mass of molecules.
        If 'allatom', calculates for all atoms in system. Defaults to 'com'.

    :param one_d (bool): If True, function also outputs Dictionary containing 1D msd data. Defaults to False.

    :param mol_species (str? Dict?): Currently unused. Will control which molecules are considered for msd calculation in conjunction w/ msd_type (probably will use 'atom').

    :param initial_timestep (int): Initial timestep for reference positions. Defaults to 0.

    :param output (bool): If True, will print the timestep of the current frame being processed. Recommended if msd_type is 'com' and system is large. Defaults to False.

    :return msd [one_d = False] (Dict): dictionary w/ keys of time (s) and values of msd data (cm^2).

    :return: (msd, msd_1D) [one_d = True] (tuple): tuple of dictionaries. msd is same as above. msd_1D is dictionary w/ keys of time (s) and values of lists of x, y, and z msd data (cm^2)
    '''
    msd_1D = {}
    msd = {}

    # # Read dump file(s), check if first element of list is the earliest dump
    Dumps = list(parse_lammps_dumps(file_pattern))
    assert(Dumps[0].timestep == initial_timestep), 'First frame in list is not the initial frame. Most likely caused by poor choice of file_pattern or incorrect choice of initial_timestep.'

    for dump in Dumps:
        if output:
            print('Processing timestep number ' + str(dump.timestep))

        # # Sorting DataFrame by atom id
        assert('id' in dump.data.columns), "Missing atom id's in dump file."
        dump.data = dump.data.sort_values(by=['id'])
        dump.data.reset_index(inplace=True)

        # # Ensuring access to unwrapped coordinates, may want to add scaled coordinates in future ('sx', 'sxu', etc.)
        if 'xu' and 'yu' and 'zu' not in dump.data.columns:
            assert ('x' and 'y' and 'z' in dump.data.columns), 'Missing wrapped and unwrapped coordinates (x y z xu yu zu)'
            assert ('ix' and 'iy' and 'iz' in dump.data.columns), 'Missing unwrapped coordinates (xu yu zu) and box location (ix iy iz) for converting wrapped coordinates (x y z) into unwrapped coordinates.'
            box_x_length = dump.box.bounds[0][1] - dump.box.bounds[0][0]
            box_y_length = dump.box.bounds[1][1] - dump.box.bounds[1][0]
            box_z_length = dump.box.bounds[2][1] - dump.box.bounds[2][0]
            dump.data['xu'] = dump.data['x'].add(dump.data['ix'].multiply(box_x_length))
            dump.data['yu'] = dump.data['y'].add(dump.data['iy'].multiply(box_y_length))
            dump.data['zu'] = dump.data['z'].add(dump.data['iz'].multiply(box_z_length))

        # # Calculating msd for all atoms in the system
        if msd_type='allatom':
            # # Making square displacement data
            dump.data[['dx2', 'dy2', 'dz2']] = dump.data[['xu', 'yu', 'zu']].subtract(Dumps[0].data[['xu', 'yu', 'zu']],axis=1).pow(2)
            dump.data['disp2'] = dump.data[['dx2', 'dy2', 'dz2']].sum(axis=1)

            # # Computing mean square displacements and converting to cm^2 and s
            time = dump.timestep * fs_per_step * 10 ** -15
            msd_val = dump.data['disp2'].mean() * 10 ** -16
            msd[time] = msd_val
            if one_d:
                msd_x_val = dump.data['dx2'].mean() * 10 ** -16
                msd_y_val = dump.data['dy2'].mean() * 10 ** -16
                msd_z_val = dump.data['dz2'].mean() * 10 ** -16
                msd_1D[time] = [msd_x_val,msd_y_val,msd_z_val]

        elif msd_type='com':
            if mol_species == 'all':
                # # Creating initial position information of CoM of molecules in DataFrame then adding msd data to Data object
                if dump.timestep == initial_timestep:
                    init_n_mols = dump.data['mol'].max()
                    print('Number of molecules in system: ' + str(init_n_mols))
                    Initial_data = pd.concat([pd.DataFrame([i + 1], columns=['Mol']) for i in range(init_n_mols)],ignore_index=True)

                    for mol in range(init_n_mols):
                        init_dump_data = dump.data[dump.data['mol'] == mol + 1]
                        x_com = init_dump_data['xu'].multiply(init_dump_data['mass']).sum() / init_dump_data['mass'].sum()
                        y_com = init_dump_data['yu'].multiply(init_dump_data['mass']).sum() / init_dump_data['mass'].sum()
                        z_com = init_dump_data['zu'].multiply(init_dump_data['mass']).sum() / init_dump_data['mass'].sum()
                        Initial_data.at[mol, 'x_com'] = x_com
                        Initial_data.at[mol, 'y_com'] = y_com
                        Initial_data.at[mol, 'z_com'] = z_com

                    Initial_data[['dx2', 'dy2', 'dz2']] = Initial_data[['x_com', 'y_com', 'z_com']].subtract(Initial_data[['x_com', 'y_com', 'z_com']], axis=1).pow(2)
                    Initial_data['disp2'] = Initial_data[['dx2', 'dy2', 'dz2']].sum(axis=1)

                    # # Computing mean square displacements for initial step and converting to cm^2 and s
                    time = dump.timestep * fs_per_step * 10 ** -15
                    msd_val = Initial_data['disp2'].mean() * 10 ** -16
                    msd[time] = msd_val
                    if one_d:
                        msd_x_val = Initial_data['dx2'].mean() * 10 ** -16
                        msd_y_val = Initial_data['dy2'].mean() * 10 ** -16
                        msd_z_val = Initial_data['dz2'].mean() * 10 ** -16
                        msd_1D[time] = [msd_x_val, msd_y_val, msd_z_val]

                # # Creating position information of CoM of molecules for timesteps later than the first one in DataFrame then calculating MSD
                else:
                    n_mols = dump.data['mol'].max()
                    assert (n_mols == init_n_mols), 'Different frames have different numbers of molecules.'
                    Current_data = pd.concat([pd.DataFrame([i + 1], columns=['Mol']) for i in range(n_mols)],ignore_index=True)

                    for mol in range(n_mols):
                        current_dump_data = dump.data[dump.data['mol'] == mol + 1]
                        x_com = current_dump_data['xu'].multiply(current_dump_data['mass']).sum() / current_dump_data['mass'].sum()
                        y_com = current_dump_data['yu'].multiply(current_dump_data['mass']).sum() / current_dump_data['mass'].sum()
                        z_com = current_dump_data['zu'].multiply(current_dump_data['mass']).sum() / current_dump_data['mass'].sum()
                        Current_data.at[mol, 'x_com'] = x_com
                        Current_data.at[mol, 'y_com'] = y_com
                        Current_data.at[mol, 'z_com'] = z_com

                    Current_data[['dx2', 'dy2', 'dz2']] = Current_data[['x_com', 'y_com', 'z_com']].subtract(Initial_data[['x_com', 'y_com', 'z_com']], axis=1).pow(2)
                    Current_data['disp2'] = Current_data[['dx2', 'dy2', 'dz2']].sum(axis=1)

                    # # Computing mean square displacements for current step and converting to cm^2 and s
                    time = dump.timestep * fs_per_step * 10 ** -15
                    msd_val = Current_data['disp2'].mean() * 10 ** -16
                    msd[time] = msd_val

                    if one_d:
                        msd_x_val = Current_data['dx2'].mean() * 10 ** -16
                        msd_y_val = Current_data['dy2'].mean() * 10 ** -16
                        msd_z_val = Current_data['dz2'].mean() * 10 ** -16
                        msd_1D[time] = [msd_x_val, msd_y_val, msd_z_val]
    if one_d:
        return msd, msd_1D
    else:
        return msd



def get_diff(filename):
    """
    This function fits the saved msd data with linear regression to calculate diffusion coefficient in cm^2/s.
    :param filename: name of file containing time (fs) and msd (Angstrom^2)
    :return: diffusion coefficient (cm^2/s)
    """
    msd = get_msd(filename)
    time_new = [*msd]
    time_array = np.array(time_new).reshape(-1, 1)
    disp_array = np.transpose(np.array(list(msd.values())))
    for d in disp_array:
        model = LinearRegression().fit(time_array, d)
        slope = model.coef_ /6
        intercept = model.intercept_
        r_sq = model.score(time_array, d)
        print('coefficient of determination:', r_sq)
        print('intercept:', intercept)
        print('diffusion coefficient:', slope)
        plt.plot(np.transpose(time_array)[0], d, color='g')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)
    plt.xlabel('time (s)')
    plt.ylabel('msd ($cm^2$)')
    plt.savefig('MSD'+'.eps', format='eps', dpi=1000)
    plt.show()
    plt.close()
    #TODO: add name of msd before display

