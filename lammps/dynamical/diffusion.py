import os
import re
import glob
import pandas as pd
import statsmodels.api as sm

from pymatgen.io.lammps.outputs import parse_lammps_log, parse_lammps_dumps

__author__ = 'Rasha Atwi, Matthew Bliss'
__version__ = '0.3'
__email__ = 'rasha.atwi@tufts.edu, Matthew.Bliss@tufts.edu'
__date__ = 'Apr 7, 2020'


def get_msd_from_dump(dump_pattern, dt=1, tao_coeff=4, msd_type='com',
                      num_mols=None, num_atoms_per_mol=None, mass=None,
                      return_all=False, com_drift=False, avg_interval=False,
                      working_dir=None):
    # TODO: cleanup; things are not being returned in a systematic way
    if not working_dir:
        working_dir = os.getcwd()
    dt = dt * 10 ** -3
    dump_files = os.path.join(working_dir, dump_pattern)

    # # Read dump file(s), check if first element of list is the earliest dump
    dumps = parse_lammps_dumps(dump_files)

    msd_dfs = []
    for dump in dumps:
        print('Processing timestep number ' + str(dump.timestep))

        # # Sorting DataFrame by atom id
        assert ('id' in dump.data.columns), "Missing atom id's in dump file."

        dump.data = dump.data.sort_values(by=['id'])
        dump.data.reset_index(inplace=True)

        # # Ensuring access to unwrapped coordinates, may want to add scaled
        # coordinates in future ('sx', 'sxu', etc.)
        if 'xu' and 'yu' and 'zu' not in dump.data.columns:
            assert ('x' and 'y' and 'z' in dump.data.columns), \
                'Missing wrapped and unwrapped coordinates (x y z xu yu zu)'
            assert ('ix' and 'iy' and 'iz' in dump.data.columns), \
                'Missing unwrapped coordinates (xu yu zu) and box location (' \
                'ix iy iz) for converting wrapped coordinates (x y z) into ' \
                'unwrapped coordinates. '
            box_x_length = dump.box.bounds[0][1] - dump.box.bounds[0][0]
            box_y_length = dump.box.bounds[1][1] - dump.box.bounds[1][0]
            box_z_length = dump.box.bounds[2][1] - dump.box.bounds[2][0]
            dump.data['xu'] = dump.data['x'].add(
                dump.data['ix'].multiply(box_x_length))
            dump.data['yu'] = dump.data['y'].add(
                dump.data['iy'].multiply(box_y_length))
            dump.data['zu'] = dump.data['z'].add(
                dump.data['iz'].multiply(box_z_length))

        # # Calculating msd for all atoms in the system
        if msd_type == 'allatom':
            df = dump.data
            df['Time (ps)'] = dump.timestep * dt
            msd_dfs.append(df)
            id_cols = ['id']
            col_1d = ['Time (ps)']

        elif msd_type == 'com':
            df = _define_mol_cols(dump, num_mols, num_atoms_per_mol, mass)
            df['Time (ps)'] = dump.timestep * dt
            msd_dfs.append(df)
            id_cols = ['type', 'mol_id']
            col_1d = ['Time (ps)', 'type']

    msd_df = pd.concat(msd_dfs).set_index(['Time (ps)'] + id_cols).sort_index()
    if com_drift:
        msd_df = _modify_dump_coordinates(msd_df)
    coords = ['xu', 'yu', 'zu']
    disps = ['dx2', 'dy2', 'dz2']
    msd_all = msd_df.copy()
    ref_df = msd_all.xs(0, 0)

    msd_all[disps] = (msd_all[coords] - msd_all[[]].join(ref_df[coords])) ** 2
    msd_all['msd'] = msd_all[disps].sum(axis=1)
    one_d_cols = disps + ['msd']
    msd = msd_all.groupby(col_1d)[one_d_cols].mean()
    if msd_type == 'com':
        msd = msd.reset_index().pivot(columns='type', index='Time (ps)')
        msd = msd.sort_index(axis=1, level=1)
        msd.columns = ["_".join([str(i) for i in v])
                       for v in msd.columns.values]
    msd.to_csv(os.path.join(working_dir, "msd.csv"))
    output = (msd,)
    if return_all:
        msd_all.to_csv(os.path.join(working_dir, "msd_all.csv"))
        output += (msd_all,)
    if avg_interval:
        times = list(msd_df.index.levels[0])
        times = times[::tao_coeff]
        msd_df = msd_df[msd_df.index.get_level_values(0).isin(times)]
        msd_df[disps] = msd_df[coords].groupby(id_cols).transform(
            lambda x: (x - x.shift(1)) ** 2)
        msd_df.drop(0, level=0)
        msd_df['msd'] = msd_df[disps].sum(axis=1)
        msd_df = msd_df.groupby(id_cols).mean()
        msd_df.to_csv(os.path.join(working_dir, "msd_int.csv"))
        output += (msd_df,)
    return output


def get_msd_from_log(log_pattern, dt=1, save_msd=False, working_dir=None):
    if not working_dir:
        working_dir = os.getcwd()
    log_files = os.path.join(working_dir, log_pattern)
    # based on pymatgen.io.lammps.outputs.parse_lammps_dumps function
    files = glob.glob(log_files)
    if len(files) > 1:
        pattern = r"%s" % log_pattern.replace("*", "([0-9]+)")
        pattern = '.*' + pattern.replace("\\", "\\\\")
        files = sorted(files, key=lambda f: int(re.match(pattern, f).group(1)))
    logs = []
    for file in files:
        logs.append(parse_lammps_log(file)[0])
    for p, l in enumerate(logs[:-1]):
        logs[p] = l[:-1]
    full_log = pd.concat(logs, ignore_index=True)
    msd = full_log.filter(regex='msd')
    # assumes columns are named as, e.g. Mgmsd[4]
    msd.columns = [i.split("msd")[0] + ' MSD ($\AA^2$)' for i in msd.columns]
    msd["Time (ps)"] = full_log["Step"] * dt * 10 ** -3  # time in ps
    # msd.swaplevel("Time (ps)", 0, 1)
    if save_msd:
        msd.to_csv(os.path.join(working_dir, "msd.csv"))
    return msd


def get_diff(msd, initial_time=None, final_time=None, dimension=3,
             working_dir=None):
    # assumes real units are used, msd.csv in ps and Angstrom2, diff in m2/s
    if initial_time is None:
        initial_time = {}
    if final_time is None:
        final_time = {}
    msd.reset_index(inplace=True)
    min_t = min(msd['Time (ps)'])
    max_t = max(msd['Time (ps)'])
    diff = []
    models = []
    counter = 0
    for col in msd.columns:
        if "msd" in col.lower():
            msd_df = \
                msd[(msd["Time (ps)"] >= initial_time.get(counter, min_t)) &
                    (msd["Time (ps)"] <= final_time.get(counter, max_t))]
            counter += 1
            model = sm.OLS(msd_df[col], msd_df["Time (ps)"]).fit()
            models.append(model)
            diff.append([model.params[0] / (2 * dimension * (10 ** 8)),
                         model.bse[0] / (2 * dimension * (10 ** 8)),
                         model.rsquared])
            col_name = col.lower()
            filename = col_name.split(" msd")[0]
            with open(
                    os.path.join(working_dir,
                                 "{}_diff.txt".format(filename)), 'w') as f:
                f.write(str(model.summary()))

    diffusion = pd.DataFrame(diff,
                             columns=["diffusion ($m^2$/s)", "std", "$R^2$"],
                             index=[col.split(" msd")[0] for col in msd.columns
                                    if "msd" in col.lower()]).T
    diffusion.to_csv(os.path.join(working_dir, "diffusion.csv"))
    print("Diffusion results written to a .csv file.")
    return diffusion, models


def _define_mol_cols(dump, num_mols=None, num_atoms_per_mol=None, mass=None):
    """
    Calculates the center of mass of each molecule.
    """
    # TODO: include this in a common module and call it from there and check
    # why I have another one in the rdf code
    mol_types = []
    mol_ids = []

    for mol_type, number_of_mols in enumerate(num_mols):
        for mol_id in range(number_of_mols):
            for atom_id in range(num_atoms_per_mol[mol_type]):
                mol_types.append(mol_type + 1)
                mol_ids.append(mol_id + 1)
    if not mass:
        assert ("mass" in dump.data.columns), \
            "Missing atom masses in dump file."
        df = pd.DataFrame(dump.data[['id', 'type', 'xu', 'yu', 'zu', 'mass']],
                          columns=["id", "type", "xu", "yu", "zu", 'mass'])
    else:
        df = pd.DataFrame(dump.data[['id', 'type', 'xu', 'yu', 'zu']],
                          columns=["id", "type", "xu", "yu", "zu"])
        df['mass'] = df.apply(lambda x: mass[int(x.type - 1)], axis=1)
    df['mol_type'] = mol_types
    df['mol_id'] = mol_ids
    df = df.drop(['type', 'id'], axis=1).set_index(['mol_type', 'mol_id'])
    mol_df = df.groupby(['mol_type', 'mol_id']). \
        apply(
        lambda x: pd.Series(x['mass'].values @ x[['xu', 'yu', 'zu']].values /
                            x.mass.sum(), index=['xu', 'yu', 'zu']))
    mol_df['mass'] = df.groupby(['mol_type', 'mol_id'])['mass'].sum()
    return mol_df.reset_index().rename(columns={'mol_type': 'type'})


def _calculate_type_com(df, ind):
    return df.groupby(ind).apply(
        lambda x: pd.Series(x['mass'].values @ x[['xu', 'yu', 'zu']].values /
                            x.mass.sum(), index=['xu', 'yu', 'zu']))


def _modify_dump_coordinates(msd_df):
    ref_com = _calculate_type_com(msd_df.xs(0, 0), ['type'])
    com = _calculate_type_com(msd_df, ['Time (ps)', 'type'])
    drift = com - com[[]].join(ref_com)
    msd_df.loc[:, ['xu', 'yu', 'zu']] -= drift
    return msd_df


def _find_linear_part():
    pass


def plot_msd_log():
    pass


def plot_msd():
    pass


def plot_diffusion():
    pass
