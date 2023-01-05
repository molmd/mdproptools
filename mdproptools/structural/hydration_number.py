import os
import time

import multiprocessing as mp
import numpy as np
import pandas as pd

from rdf_cn import _calc_rsq, _calc_atom_type

from pymatgen.io.lammps.outputs import parse_lammps_dumps


def get_angle(data_head, water_df, r_cut, lx, ly, lz):
    data_head = data_head[['mol_id', 'x', 'y', 'z']].values
    data_i, rsq = \
        _calc_rsq(data_head, water_df.loc[:, ['mol_id', 'x', 'y', 'z']].values,
                  lx, ly, lz, 1)
    cond = rsq < r_cut ** 2
    data_i = data_i[cond]
    final_df = pd.DataFrame(data_i[:, :4],
                            columns=['mol_id', 'x', 'y', 'z'])
    final_df = final_df. \
        merge(water_df.loc[cond, ['mol_id', 'x_v', 'y_v', 'z_v']],
              on='mol_id')
    a = final_df[['x', 'y', 'z']].values * final_df[
        ['x_v', 'y_v', 'z_v']].values
    dot_prod = np.sum(a, axis=1)
    norm_1 = np.linalg.norm(final_df[['x', 'y', 'z']].values, axis=1)
    norm_2 = np.linalg.norm(final_df[['x_v', 'y_v', 'z_v']].values, axis=1)
    cos = dot_prod / (norm_1 * norm_2)
    # print([np.rad2deg(np.arccos(i)) for i in cos])
    return list(cos), len(cos[cos < -0.72]) / len(cos)


def get_all_angles(counter, dump, alter_atom_ids, num_mols, num_atoms_per_mol,
                   cation_type, water_type, r_cut):
    print(counter)
    lx = dump.box.bounds[0][1] - dump.box.bounds[0][0]
    ly = dump.box.bounds[1][1] - dump.box.bounds[1][0]
    lz = dump.box.bounds[2][1] - dump.box.bounds[2][0]
    df = dump.data.sort_values(by=['id'])
    if alter_atom_ids:
        data = _calc_atom_type(df.values, num_mols, num_atoms_per_mol)
        df["type"] = data[:, 0]
    mol_types = []
    mol_ids = []
    for mol_type, number_of_mols in enumerate(num_mols):
        for mol_id in range(number_of_mols):
            for atom_id in range(num_atoms_per_mol[mol_type]):
                mol_types.append(mol_type + 1)
                mol_ids.append(mol_id + 1)
    df['mol_type'] = mol_types
    df['mol_id'] = mol_ids
    cations_df = df.loc[df["mol_type"] == cation_type]
    water_df = df.loc[df["mol_type"] == water_type]
    cations_df = cations_df.loc[:, cations_df.columns.intersection(
        ["mol_id", "x", "y", "z"])]
    water_df = water_df.loc[:, water_df.columns.intersection(
        ["mol_id", "x", "y", "z"])]
    water_df_coord = water_df.groupby(['mol_id']).first()

    water_df_vec = water_df.groupby(['mol_id']).nth([1, 2]). \
                       groupby(['mol_id']).sum() - 2 * water_df_coord
    water_df = water_df_coord. \
        join(water_df_vec.
             rename(columns={'x': 'x_v', 'y': 'y_v', 'z': 'z_v'})). \
        reset_index()
    cosines = []
    factor_ = 0
    for _, i in cations_df.iterrows():
        cos, f = get_angle(i, water_df, r_cut, lx, ly, lz)
        cosines += cos
        factor_ += f
        # break
    return cosines, factor_ / cations_df.shape[0]


def get_hydration_number(dump_pattern, cation_type, water_type, r_cut,
                         alter_atom_ids=False, num_mols=None,
                         num_atoms_per_mol=None, working_dir=None):
    if not working_dir:
        working_dir = os.getcwd()
    st = time.time()
    dumps = parse_lammps_dumps(os.path.join(working_dir, dump_pattern))
    print(time.time() - st)
    p = mp.Pool(mp.cpu_count())
    itr = [[i, dump, alter_atom_ids, num_mols, num_atoms_per_mol, cation_type,
            water_type, r_cut] for i, dump in enumerate(dumps)]
    res = p.starmap(get_all_angles, itr)
    # res = []
    # for i, dump in enumerate(dumps):
    #     res.append(get_all_angles(i, dump, alter_atom_ids, num_mols, num_atoms_per_mol,
    #                     cation_type, water_type, r_cut))
    #     break
    angles_df = pd.DataFrame([item for sublist in res for item in sublist[0]],
                             columns=["angles_distribution"])
    angles_df["hydration_factor"] = sum([i[1] for i in res]) / len(res)
    angles_df.to_csv(os.path.join(working_dir, "angles_df.csv"))
    return angles_df
