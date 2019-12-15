from timeit import default_timer as timer
from time import time
import math

import pandas as pd
import numpy as np
from numba import njit
from pymatgen.io.lammps.outputs import parse_lammps_dumps

CON_CONSTANT = 1.660538921


@njit(cache=True)
def _calc_rsq(data_head, mol_data, lx, ly, lz):
    data_i = np.zeros((mol_data.shape[0], mol_data.shape[1] + 2))
    data_i[:, :4] = mol_data.copy()
    # data_i = np.zeros((mol_data.shape[0] - i - 1, data.shape[1] + 2))
    # data_i[:, :-2] = data[i + 1:, :]
    data_i[:, 1:4] = data_head[1:] - mol_data[:, 1:]
    dx = data_i[:, 1]
    dy = data_i[:, 2]
    dz = data_i[:, 3]
    cond = (dx > lx / 2) | (dx < -lx / 2)
    dx[cond] = dx[cond] - np.sign(dx[cond]) * lx
    cond = (dy > ly / 2) | (dy < -ly / 2)
    dy[cond] = dy[cond] - np.sign(dy[cond]) * ly
    cond = (dz > lz / 2) | (dz < -lz / 2)
    dz[cond] = dz[cond] - np.sign(dz[cond]) * lz
    rsq = dx ** 2 + dy ** 2 + dz ** 2
    data_i[:, 4] = rsq
    return data_i, rsq


@njit(cache=True)
def _main_loop(atom_data, mol_data, relation_matrix, n_a_pairs,
               lengths, rcut, ddr, rdf_full, rdf_p):
    """
    This function saves the data into a 2d array with column (0) as the type,
    columns (1, 2, 3) as x, y, and z, column (4) as rsq
    and column (5) as the bin number. The main for loop is compiled before
    the code is executed.
    """
    lx = lengths[0]
    ly = lengths[1]
    lz = lengths[2]
    for i in range(0, atom_data.shape[0] - 1):
        data_head = atom_data[i, :]
        data_i, rsq = _calc_rsq(data_head, mol_data, lx, ly, lz)
        cond = rsq < rcut ** 2
        data_i = data_i[cond, :]
        data_i[:, 5] = np.sqrt(data_i[:, 4]) / ddr
        for j in data_i[:, -1].astype(np.int64):
            rdf_full[j] += 1
        for kl in range(0, n_a_pairs):
            nta1, nta2 = relation_matrix[kl]
            if int(data_head[0]) == nta1:
                v_j = data_i[data_i[:, 0].astype(np.int64) == nta2]
                for j in v_j[:, -1].astype(np.int64):
                    rdf_p[kl][j] += 1
    return rdf_full, rdf_p


@njit(cache=True)
def _cn_loop(atom_data, mol_data, relation_matrix, n_a_pairs,
               lengths, ddr, cn):
    """
    This function saves the data into a 2d array with column (0) as the type,
    columns (1, 2, 3) as x, y, and z, column (4) as rsq
    and column (5) as the bin number. The main for loop is compiled before
    the code is executed.
    relation_matrix = [[ref_atom], [pair_atom], [r_cut]]
    """
    lx = lengths[0]
    ly = lengths[1]
    lz = lengths[2]
    for i in range(0, atom_data.shape[0] - 1):
        data_head = atom_data[i, :]
        data_i, rsq = _calc_rsq(data_head, mol_data, lx, ly, lz)
        for kl in range(0, n_a_pairs):
            nta1, nta2, rcut = relation_matrix[kl]
            cond = rsq < rcut ** 2
            data_i = data_i[cond, :]
            data_i[:, 5] = np.sqrt(data_i[:, 4]) / ddr
            if int(data_head[0]) == nta1:
                v_j = data_i[data_i[:, 0].astype(np.int64) == nta2]
                cn[kl] += len(v_j[:, -1])
    return cn


def _define_mol_cols(df, nmols, natoms_per_mol, mass):
    mol_types = []
    mol_ids = []
    for mol_type, number_of_mols in enumerate(nmols):
        for mol_id in range(number_of_mols):
            for atom_id in range(natoms_per_mol[mol_type]):
                mol_types.append(mol_type + 1)
                mol_ids.append(mol_id + 1)
    df['mol_type'] = mol_types
    df['mol_id'] = mol_ids
    df['mass'] = df.apply(lambda x: mass[int(x.type - 1)], axis=1)
    df = df.drop(['type'], axis=1).set_index(['mol_type', 'mol_id'])
    mol_df = df.groupby(['mol_type', 'mol_id']).\
        apply(lambda x: pd.Series(x['mass'].values@x[['x', 'y', 'z']].values/
                                  x.mass.sum(), index=['x', 'y', 'z']))
    return mol_df.reset_index().drop('mol_id', axis=1)


def _calc_com(mol_df):
    mol_df = mol_df.groupby(['mol_type', 'mol_id']).\
        apply(lambda x: pd.Series(x['mass'].values@x[['x', 'y', 'z']].values/
                                  x.mass.sum(), index=['x', 'y', 'z']))
    return mol_df.reset_index().drop('mol_id', axis=1)


def calc_molecular_rdf(r_cut, bin_size, ntypes, mass, n_part_rdfs,
                       relation_matrix_og, filename, nmols, natoms_per_mol):

    # may want to write function to ensure that r_cut is a multiple of bin_size
    n_bins = int(r_cut/bin_size)
    rdf_full_sum = np.zeros(n_bins)
    rdf_part_sum = np.zeros((n_part_rdfs, n_bins))


    dumps = list(parse_lammps_dumps(filename))
    n_files = len(dumps)
    radii = (np.arange(n_bins) + 0.5) * bin_size

    for dump in dumps:
        start_traj_loop = timer()
        print('The timestep of the current file is: ' + str(dump.timestep))
        df = dump.data[['id', 'type', 'x', 'y', 'z']]
        df = df.sort_values('id').drop('id', axis=1)
        mol_df = df.copy()
        if len(mol_df) != sum([i*j for i, j in zip(nmols, natoms_per_mol)]):
            raise ValueError('The number of atoms is inconsistent')
        mol_df = _define_mol_cols(mol_df, nmols, natoms_per_mol, mass)
        n_atoms = df.shape[0]
        n_mols = mol_df.shape[0]
        box_lengths = dump.box.to_lattice().lengths
        volume = np.prod(box_lengths)
        atom_types = df.type.astype(np.int64).value_counts().to_dict()
        mol_types = mol_df.mol_type.astype(np.int64).value_counts().to_dict()
        set_id = set(atom_types.keys())
        if ntypes != len(set_id):
            raise Exception(f"""Consistency check failed:
                        Number of atomic types in the config file is
                        different from the corresponding value in input file
                        ntypes=: {ntypes}, nset=: {len(set_id)}""")
        total_mass = np.sum([float(mass[i]) * float(atom_types[i + 1])
                        for i in range(ntypes)])
        print(total_mass)
        total_density = float((total_mass / volume) * CON_CONSTANT)
        print(total_density)
        print('{0:s}{1:10.8f}'.format('Average density=:',
                                      float(total_density)))

        rho = n_mols / volume

        rho_n_pairs = np.zeros(n_part_rdfs)
        for index, mol_type in enumerate(relation_matrix_og[1]):
            rho_n_pairs[index] = mol_types[mol_type] / volume
            if rho_n_pairs[index] < 1.0e-22:
                raise Exception('Error: Density is zero for mol type: ' +
                                str(mol_type))
        relation_matrix = np.asarray(relation_matrix_og).transpose()
        rdf_full = np.zeros(n_bins)
        rdf_part = np.zeros((n_part_rdfs, n_bins))
        st = time()
        atom_data = df.values
        mol_data = mol_df.values
        rdf_full, rdf_part = \
            _main_loop(atom_data, mol_data, relation_matrix, n_part_rdfs,
                       box_lengths, r_cut, bin_size, rdf_full, rdf_part)
        print("time:", time() - st)
        print("Finished computing RDF for timestep", dump.timestep)

        # Normalization Procedure for the full RDF and partical RDFs
        shell_volume = 4 / 3 * math.pi * bin_size ** 3 * \
                       (np.arange(1,n_bins+1) ** 3 - np.arange(n_bins) ** 3)
        rdf_full = rdf_full / (n_mols * rho * shell_volume)

        ref_atoms = np.asarray(relation_matrix_og[0]).reshape((n_part_rdfs, 1))
        ref_atoms_matrix = np.tile(ref_atoms, n_bins)
        n_atoms_matrix = np.vectorize(atom_types.get)(ref_atoms_matrix)
        rho_n_pairs_matrix = \
            np.tile(rho_n_pairs.reshape((n_part_rdfs, 1)), n_bins)

        shell_volume_matrix = np.tile(shell_volume, (n_part_rdfs, 1))

        rdf_part = rdf_part / (n_atoms_matrix * rho_n_pairs_matrix *
                               shell_volume_matrix)
        rdf_full_sum = rdf_full_sum + rdf_full
        rdf_part_sum = rdf_part_sum + rdf_part

        end_traj_loop = timer()
        print('Trajectory loop took:', end_traj_loop - start_traj_loop, 's')

    rdf_full_sum = rdf_full_sum / n_files
    rdf_part_sum = rdf_part_sum / n_files

    final_data_array = np.vstack((radii, rdf_full_sum, rdf_part_sum)).\
        transpose()
    final_data_labels = ['r [Angst]', 'g_full(r)'] + \
                        ['g_' + str(pair[0]) + '-' + str(pair[1]) + '(r)'
                         for pair in np.asarray(relation_matrix_og).transpose()]
    final_data_frame = pd.DataFrame(final_data_array, columns=final_data_labels)
    final_data_frame.to_csv(path_or_buf='rdf.csv', index=False)

    print("Full RDF and partial RDFs are written to rdf.csv file")
    return final_data_frame


if __name__ == '__main__':
    c_file_name = '/Users/rashaatwi/Desktop/Mg/Mg_2TFSI_G1.lammpstrj.*'
    rasha = calc_molecular_rdf(20, 0.05, 8,
                               [16, 12.01, 1.008, 14.01, 32.06, 16, 19, 24.305],
                               3, [[8, 8,  4], [1, 2, 3]], c_file_name,
                               [591, 66, 33], [16, 15, 1])
    print(rasha)





