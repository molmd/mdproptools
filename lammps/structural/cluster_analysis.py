import os
import shutil
import glob

import numpy as np
import pandas as pd

from rdf_cn import _calc_rsq, _calc_atom_type

from pymatgen.io.lammps.outputs import parse_lammps_dumps
from pymatgen.core.structure import Molecule
from pymatgen.analysis.molecule_matcher import MoleculeMatcher

"""
This module extracts clusters from LAMMPS trajectory files and groups them 
into unique configurations.
"""

__author__ = "Rasha Atwi, Maxim Makeev"
__version__ = "0.1.1"
__email__ = "rasha.atwi@tufts.edu, maxim.makeev@tufts.edu"
__date__ = "Apr 28, 2020"

FORCE_CONSTANT = 0.043363 / 16.0


def _remove_boundary_effects(data_head, mol_data, lx, ly, lz, num_of_ids):
    data_i = np.zeros((mol_data.shape[0], mol_data.shape[1]))
    data_i[:, :num_of_ids + 3] = mol_data.copy()
    dxyz = mol_data[:, num_of_ids:] - data_head[num_of_ids:]
    dx = dxyz[:, 0]
    dy = dxyz[:, 1]
    dz = dxyz[:, 2]
    cond = (dx > lx / 2) | (dx < -lx / 2)
    data_i[cond, num_of_ids] = data_i[cond, num_of_ids] - np.sign(dx[cond]) * lx
    cond = (dy > ly / 2) | (dy < -ly / 2)
    data_i[cond, num_of_ids + 1] = data_i[cond, num_of_ids + 1] - np.sign(
        dy[cond]) * ly
    cond = (dz > lz / 2) | (dz < -lz / 2)
    data_i[cond, num_of_ids + 2] = data_i[cond, num_of_ids + 2] - np.sign(
        dz[cond]) * lz
    return data_i


def get_clusters(dump_pattern, frame, atom_type, r_cut, num_mols=None,
                 num_atoms_per_mol=None, elements=None, alter_atom_ids=False,
                 max_force=0.75, working_dir=None):
    if elements:
        elements = {i + 1: j for i, j in enumerate(elements)}
    if not working_dir:
        working_dir = os.getcwd()
    dumps = list(parse_lammps_dumps(os.path.join(working_dir, dump_pattern)))
    dump = dumps[frame]
    lx = dump.box.bounds[0][1] - dump.box.bounds[0][0]
    ly = dump.box.bounds[1][1] - dump.box.bounds[1][0]
    lz = dump.box.bounds[2][1] - dump.box.bounds[2][0]
    df = dump.data.sort_values(by=['id'])
    mol_types = []
    mol_ids = []
    for mol_type, number_of_mols in enumerate(num_mols):
        for mol_id in range(number_of_mols):
            for atom_id in range(num_atoms_per_mol[mol_type]):
                mol_types.append(mol_type + 1)
                mol_ids.append(mol_id + 1)
    df['mol_type'] = mol_types
    df['mol_id'] = mol_ids
    if elements:
        df["element"] = df["type"].map(elements)
    if alter_atom_ids:
        data = _calc_atom_type(df.values, num_mols, num_atoms_per_mol)
        df["type"] = data[:, 0]
    counter = 0
    for i in df[df["type"] == atom_type]["id"]:
        counter += 1
        print("Processing atom number: {}".format(counter - 1))
        data_head = \
            df[df['id'] == i][['mol_type', 'mol_id', 'x', 'y', 'z']].values[0]
        data_i, rsq = _calc_rsq(data_head, df.loc[:,
                                           ['mol_type', 'mol_id', 'x', 'y',
                                            'z']].values, lx, ly, lz, 2)
        cond = rsq < r_cut ** 2
        data_i = data_i[cond, :]
        ids = pd.DataFrame(np.unique(data_i[:, [0, 1]], axis=0),
                           columns=['mol_type', 'mol_id'])
        neighbor_df = ids.merge(df, on=['mol_type', 'mol_id'])
        min_force = neighbor_df.groupby(['mol_type', 'mol_id']). \
                        agg({'fx': 'sum', 'fy': 'sum', 'fz': 'sum'}). \
                        min(axis=1) * FORCE_CONSTANT
        min_force_atoms = min_force[min_force < max_force]. \
            reset_index()[['mol_type', 'mol_id']]. \
            merge(neighbor_df, on=['mol_type', 'mol_id'])
        # if elements:
        #     min_force_atoms['element'] = min_force_atoms['type'].map(elements)
        data_head = \
            df[df['id'] == i][['id', 'x', 'y', 'z']].values[0]
        data_i = min_force_atoms.loc[min_force_atoms['id'] != i,
                                     ['id', 'x', 'y', 'z']].values
        data_i = np.insert(data_i, 0, data_head, axis=0)
        data_i = _remove_boundary_effects(data_head, data_i, lx, ly, lz, 1)
        fin_df = pd.DataFrame(data_i, columns=['id', 'x', 'y', 'z']). \
            merge(min_force_atoms[['element', 'id']], on='id'). \
            drop('id', axis=1). \
            set_index('element'). \
            reset_index()
        if frame < 10:
            frame_number = '0{}'.format(frame)
        else:
            frame_number = frame
        if counter - 1 < 10:
            filename = 'Cluster_{}_0{}.xyz'.format(frame_number, counter - 1)
        else:
            filename = 'Cluster_{}_{}.xyz'.format(frame_number, counter - 1)
        f = open((os.path.join(working_dir, filename)), 'a')
        f.write('{}\n\n'.format(len(fin_df)))
        fin_df.to_csv(f, header=False, index=False, sep='\t',
                      float_format='%15.10f')
        f.close()
    print("{} clusters written to *.xyz files".
          format(len(df[df["type"] == atom_type]["id"])))


def group_clusters(cluster_pattern, tolerance=0.1, working_dir=None):
    if not working_dir:
        working_dir = os.getcwd()
    mm = MoleculeMatcher(tolerance=tolerance)
    filename_list = glob.glob((os.path.join(working_dir, cluster_pattern)))
    print(filename_list)
    mol_list = [Molecule.from_file(os.path.join(working_dir, f)) for
                f in filename_list]
    mol_groups = mm.group_molecules(mol_list)
    filename_groups = [[filename_list[mol_list.index(m)] for m in g] for g in
                       mol_groups]
    for p, i in enumerate(filename_groups):
        if p+1 < 10:
            conf_num = "0{}".format(p+1)
        else:
            conf_num = p+1
        folder_name = "Configuration_{}".format(conf_num)
        if not os.path.exists(folder_name):
            os.mkdir((os.path.join(working_dir, folder_name)))
            for f in i:
                shutil.move(f, (os.path.join(working_dir, folder_name)))
        else:
            for f in i:
                shutil.move(f, (os.path.join(working_dir, folder_name)))

