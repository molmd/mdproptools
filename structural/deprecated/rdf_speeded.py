
# !/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains files to calculate full and partial rdfs from LAMMPS trajectories

# __author__ = 'Rasha Atwi, Maxim Makeev, Matthew Bliss'
# __version__ = '0.4'
# __email__ = 'rasha.atwi@tufts.edu, maxim.makeev@tufts.edu, matthew.bliss@tufts.edu'
# __date__ = 'Oct 9, 2019'

import os
import sys
import math

import numpy as np
import pandas as pd
from numba import jit, njit
from time import time
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from timeit import default_timer as timer

@njit(cache=True)
def main_loop(data, relation_matrix, n_a_pairs, Lengths, rcut, ddr, RDF_FULL, RDF_P):
    """
    This function calculates full and partial rdf based on LAMMPS dump files. Currently only calculates the partial rdfs
    based on LAMMPS atom types. Works for both single and multiple LAMMPS dump files. The main for loop is compiled
    before the code is executed.
    :param data: (array-like) LAMMPS dump data as a 2d array with column (0) as the id,
    column (2) as type, and columns (3, 4, 5) as x, y, and z
    :param relation_matrix: (array-like) 2d array with column (0) as the first atom types and column (1)
    as the corresponding atom types for which the partial RDFs are computed
    :param n_a_pairs: (int) The number of partial rdfs to calculate
    :param Lengths: (list) Box dimensions in x, y, z directions
    :param rcut: (float or int) Maximum radius for which rdf is calculated in LAMMPS units
    :param ddr: (float) Bin size of the histogram in LAMMPS units
    :param RDF_FULL: (array-like) 1d array of zeros of size of the number of bins
    :param RDF_P: (array-like) 2d array of zeros of size of number of partial rdfs by number of bins
    :return: (array-like) RDF_Full: 1d array with computed full rdf
    :return: (array-like) RDF_P: 2d array with computed partial rdf between each atom pair
    """
    Lx = Lengths[0]
    Ly = Lengths[1]
    Lz = Lengths[2]
    for i in range(0, data.shape[0] - 1):
        data_head = data[i, :]
        data_i = np.zeros((data.shape[0] - i - 1, data.shape[1] + 2))
        data_i[:, :-2] = data[i + 1:, :]
        data_i[:, 2:5] = data_head[2:] - data_i[:, 2:5]
        dx = data_i[:, 2]
        dy = data_i[:, 3]
        dz = data_i[:, 4]
        cond = (dx > Lx / 2) | (dx < -Lx / 2)
        dx[cond] = dx[cond] - np.sign(dx[cond]) * Lx
        cond = (dy > Ly / 2) | (dy < -Ly / 2)
        dy[cond] = dy[cond] - np.sign(dy[cond]) * Ly
        cond = (dz > Lz / 2) | (dz < -Lz / 2)
        dz[cond] = dz[cond] - np.sign(dz[cond]) * Lz
        rsq = dx ** 2 + dy ** 2 + dz ** 2
        data_i[:, 5] = rsq
        cond = rsq < rcut ** 2
        data_i = data_i[cond, :]
        data_i[:, 6] = np.sqrt(data_i[:, 5]) / ddr
        for j in data_i[:, 6].astype(np.int64):
            RDF_FULL[j] += 2
        for kl in range(0, n_a_pairs):
            nta1, nta2 = relation_matrix[kl]
            if data_head[1] == nta1:
                v_j = data_i[data_i[:, 1] == nta2]
                for j in v_j[:, 6].astype(np.int64):
                    RDF_P[kl][j] += 1
            if data_head[1] == nta2:
                v_j = data_i[data_i[:, 1] == nta1]
                for j in v_j[:, 6].astype(np.int64):
                    RDF_P[kl][j] += 1
    return RDF_FULL, RDF_P


def calc_rdf(r_cut,bin_size,check_n_atoms,ntypes,Mass,n_part_rdfs,Atom_types,filename):
    '''
    This function calculates full and partial rdf based on LAMMPS dump files. Currently only calculates the partial rdfs
    based on LAMMPS atom types. Saves the output in a *.csv file. Works for both single and multiple LAMMPS dump files.
    :param r_cut: (float or int) Maximum radius for which rdf is calculated in LAMMPS units
    :param bin_size: (float) Bin size of the histogram in LAMMPS units.
    :param check_n_atoms: (int) Number of atoms in the dump files. Not currently used, not sure if it is necessary.
    :param ntypes: (int) The total number of LAMMPS atom types in the system
    :param Mass: (array-like) This contains the masses of all LAMMPS atom types in numerical order
    :param n_part_rdfs: (int) The number of partial rdfs to calculate
    :param Atom_types: (array-like) A 2 by N array-like with N being the same as n_part_rdfs. The first sub-array-like
        contains the LAMMPS atom types for the reference atoms, and the second contains the corresponding LAMMPS atom
        type pairs.
    :param filename: (str) Can be the entire filename for calculations involving a single file, or a file pattern with
        the wildcard character ('*').
    :return: (Pandas DataFrame) Contains the radii, full rdf, and partial rdfs. Also prints out the DataFrame and saves
        it to a *.csv file.
    '''
    ConConstant = 1.660538921
    n_bins = int(r_cut/bin_size) # may want to write function to ensure that r_cut is a multiple of bin_size
    Radii = (np.arange(n_bins) + 0.5) * bin_size
    Rdf_full_sum = np.zeros(n_bins)
    Rdf_part_sum = np.zeros((n_part_rdfs,n_bins))

    Dumps = list(parse_lammps_dumps(filename))
    n_files = len(Dumps)

    if n_files == 0:
        raise Exception('parse_lammps_dumps() did not detect any files with the filename or filepattern provided.')

    for Dump in Dumps:
        start_traj_loop = timer()
        print('The timestep of the current file is: ' + str(Dump.timestep))
        df = Dump.data[['id', 'type', 'x', 'y', 'z']]
        n_atoms = df.shape[0]
        Box_lengths = Dump.box.to_lattice().lengths
        volume = np.prod(Box_lengths)

        rho_n_pairs = np.zeros(n_part_rdfs)
        atomtypes = df.type.value_counts().to_dict()
        setID = set(atomtypes.keys())
        if ntypes != len(setID):
            raise Exception(f"""Consistency check failed:
                        Number of atomic types in the config file is
                        different from the corresponding value in input file
                        ntypes=: {ntypes}, nset=: {len(setID)}""")
        massT = sum([float(Mass[i]) * float(atomtypes[i + 1]) for i in range(ntypes)])
        densT = float((massT / volume) * ConConstant)
        print('{0:s}{1:10.8f}'.format('Average density=:', float(densT)))
        rho = n_atoms / volume
        relation_matrix = np.asarray(Atom_types).transpose()
        for index, atom_type in enumerate(Atom_types[1]):
            rho_n_pairs[index] = atomtypes[atom_type] / volume
            if rho_n_pairs[index] < 1.0e-22:
                raise Exception('Error: Density is zero for atom type: ' + str(atom_type))

        data = df.values
        Rdf_full = np.zeros(n_bins)
        Rdf_part = np.zeros((n_part_rdfs, n_bins))
        st = time()
        Rdf_full, Rdf_part = main_loop(data, relation_matrix, n_part_rdfs, Box_lengths, r_cut, bin_size, Rdf_full, Rdf_part)
        print("time:", time() - st)
        print("Finished computing RDF for timestep", Dump.timestep)

        # Normalization Procedure for the full RDF
        Shell_volume = 4 / 3 * math.pi * bin_size ** 3 * (np.arange(1,n_bins+1) ** 3 - np.arange(n_bins) ** 3)
        Rdf_full = Rdf_full / (n_atoms * rho * Shell_volume)

        # Normalization Procedure for the partial RDF
        Ref_atoms = np.asarray(Atom_types[0]).reshape((n_part_rdfs, 1))
        Ref_atoms_matrix = np.tile(Ref_atoms, n_bins)
        n_atoms_matrix = np.vectorize(atomtypes.get)(Ref_atoms_matrix)
        Rho_n_pairs_matrix = np.tile(rho_n_pairs.reshape((n_part_rdfs,1)), n_bins)
        Shell_volume_matrix = np.tile(Shell_volume, (n_part_rdfs,1))
        Rdf_part = Rdf_part / (n_atoms_matrix * Rho_n_pairs_matrix * Shell_volume_matrix)

        Rdf_full_sum = Rdf_full_sum + Rdf_full
        Rdf_part_sum = Rdf_part_sum + Rdf_part

        end_traj_loop = timer()
        print('Trajectory loop took: ' + str(end_traj_loop - start_traj_loop) + ' s')

    Rdf_full_sum = Rdf_full_sum / n_files
    Rdf_part_sum = Rdf_part_sum / n_files

    # Making DataFrame with all rdf data
    Final_data_array = np.vstack((Radii,Rdf_full_sum,Rdf_part_sum)).transpose()
    Final_data_labels = ['r [Angst]','g_full(r)'] + ['g_' + str(Pair[0]) + '-' + str(Pair[1]) + '(r)' for Pair in relation_matrix]
    Final_data_frame = pd.DataFrame(Final_data_array, columns=Final_data_labels)
    print(Final_data_frame)

    Final_data_frame.to_csv(path_or_buf='rdf.csv', index=False)
    print("Full RDF and partial RDFs are written to rdf.csv file")
    return Final_data_frame