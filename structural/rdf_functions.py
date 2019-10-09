# !/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains functions to calculate full and partial rdfs from LAMMPS trajectories, as well as some code from
# testing

# __author__ = 'Rasha Atwi, Maxim Makeev, Matthew Bliss'
# __version__ = '0.1'
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
    This function saves the data into a 2d array with column (0) as the id,
    column (2) as type, columns (3, 4, 5) as x, y, and z, column (5) as rsq
    and column (6) as the bin number. The main for loop is compiled before
    the code is executed.
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

start = timer()
ConConstant = 1.660538921
NumConstant = float(4.0 / 3.0)

# Partial RDFs are computed between types: nrdf[n] and nrdf[m]
nrdf0 = []
nrdf1 = []
mass = []

# Containers for atomic ID, type and Cartesian coordinates
currwd = os.getcwd()
print("Current directory:\n", currwd)
rdfinput = open("input.dat", "r")
AI = rdfinput.readlines()
rdfinput.close()

# # Parameters-loaded from the input file
tmp01 = AI[1].split()
r_cut = float(tmp01[0])
tmp02 = AI[3].split()
bin_size = float(tmp02[0])
tmp03 = AI[5].split()
check_n_atoms = int(tmp03[0])
tmp04 = AI[7].split()
ntypes = int(tmp04[0])
tmp05 = AI[10].split()
for kk in range(0, ntypes):
    mass.append(tmp05[kk])
tmp06 = AI[12].split()
nrdfs = int(tmp06[0])
tmp07 = AI[15].split()
tmp08 = AI[16].split()
for ll in range(0, nrdfs):
    nrdf0.append(tmp07[ll])
    nrdf1.append(tmp08[ll])
print("Compute partial RDFs:")
for l0 in range(0, nrdfs):
    print(nrdf0[l0], end=' ')
print('')
for l1 in range(0, nrdfs):
    print(nrdf1[l1], end=' ')
print('')
tmp09 = AI[19].split()
input_mode = tmp09[0]
filename = tmp09[1]
print('Filename:')
print(filename)
tmp10 = AI[21].split()
workdir0 = tmp10[0]
# workdir0 = '/Users/mbliss01/Documents/Tufts/First_Year/Research/Cascade/DHPS/SPCE/nvt/Traj/Full_Traj'
nbin = int(r_cut / bin_size) + 1
workdir = os.chdir(workdir0)
pathdir = os.getcwd()
print("Working directory:")
print(pathdir)
end = timer()
print('Reading input file:')
print(end-start)

# # This reads all dump files in the current directory, and sets them as LammpsDump objects in a generator
# start = timer()
# Dumps = list(parse_lammps_dumps(filename+'*'))
# end = timer()
# print('Parsing dump files:')
# print(str(end - start))
# # Dumps is generator
# # 3.668e-06 s for   5 files: 7.336e-07 s per file
# # 1.663e-06 s for 801 files: 2.076e-09 s per file
# # Dumps is list
# # 1.030 s     for   5 files: 0.206 s per file
# # 148.6 s     for 801 files: 0.186 s per file
#
# print(len(Dumps))
#
# # # # Testing for how long it takes to iterate over generator object of dumps
# # iter = 0
# # start = timer()
# # for Dump in Dumps:
# #     # print(Dump.timestep)
# #     iter += 1
# # end = timer()
# # print('Number of dump files:')
# # print(iter)
# # print('Iterating over dump files:')
# # print(end-start)
# # # 1.066 s for   5 files: 0.2132 s per file
# # # 148.3 s for 801 files: 0.1851 s per file
#
# Dump_0 = Dumps[0]
# print(type(Dump_0.box))
# print(type(Dump_0.box.bounds))
# x_0 = Dump_0.box.bounds[0][1] - Dump_0.box.bounds[0][0]
# y_0 = Dump_0.box.bounds[1][1] - Dump_0.box.bounds[1][0]
# z_0 = Dump_0.box.bounds[2][1] - Dump_0.box.bounds[2][0]
# print(str(x_0) + ', ' + str(y_0) + ', ' + str(z_0))
# print(Dump_0.box.to_lattice().lengths)
# print(np.prod(Dump_0.box.to_lattice().lengths))
# print(x_0 * y_0 * z_0)


# # Defined inputs
r_cut = 20
bin_size = 0.2
check_n_atoms = 35253
ntypes = 12
mass = [14.010, 12.010, 12.010, 12.010, 1.008, 16.000, 32.060, 16.000, 1.008, 16.000, 1.008, 22.990]
nrdfs = 7
atom_types = [[12, 12, 12, 12, 12, 6, 1],
              [ 1,  6,  8, 10, 12, 6, 1]]
filename = 'dump.0.5M_dhps_2.5M_na_1M_oh.nvt_298.15K.*'
# filename = 'dump.0.5M_dhps_2.5M_na_1M_oh.nvt_298.15K.0.lammpstrj'
workdir = './'
os.chdir(workdir0)

print(np.asarray(atom_types).shape)

# Ref_atoms = np.asarray(atom_types[0]).reshape((len(atom_types[0]),1))
# print(Ref_atoms)
# Ref_atoms_matrix = np.tile(Ref_atoms,int(r_cut/bin_size))
# print(Ref_atoms_matrix)
# Atom_dict = {9: 21122, 8: 10561, 2: 840, 12: 525, 6: 525, 5: 525, 11: 210, 10: 210, 4: 210, 3: 210, 1: 210, 7: 105}
# n_atoms_matrix = np.vectorize(Atom_dict.get)(Ref_atoms_matrix)
# print(n_atoms_matrix)


# nbins = int(r_cut/bin_size)
# Radius = (np.arange(nbins) + 0.5) * bin_size
# print(Radius)
#
# RDF_COOR = []
# for i in range(0, nbins):
#     rr = float(bin_size * (float(i) + float(0.5)))
#     RDF_COOR.append(rr)
#
# # print(RDF_COOR)
#
# # print(np.arange(1,nbins + 1))
# # print(np.arange(1,nbins + 1) ** 3)
# # print(np.arange(nbins) ** 3)
# SHELL_VOL = (np.arange(1,nbins + 1) ** 3 - np.arange(nbins) ** 3)
# # print(SHELL_VOL)
# # print(SHELL_VOL/(np.arange(1,nbins + 1) ** 3))
#
# Rdf_full = np.zeros((1,nbins))
# Rdf_part = np.ones((nrdfs,nbins))
# print(Rdf_full)
# print(Rdf_part)
# # print(Rdf_part,'\n')
# Check_partial = np.tile(Rdf_full,(nrdfs,1))
# # print(Check_partial)
# # print(np.array_equal(Rdf_part,Check_partial))
# # print(np.tile(RDF_COOR,(nrdfs,1)))
#
# print(np.shape(Rdf_full))
# print(np.shape(Rdf_part))
#
# Final_data_array = np.vstack((Radius,Rdf_full,Rdf_part))
# print(Final_data_array.transpose())
#
# relation_matrix = np.asarray(atom_types).transpose()
# Final_data_labels = ['r [Angst]','g_full(r)'] + ['g_' + str(Pair[0]) + '-' + str(Pair[1]) + '(r)' for Pair in relation_matrix]
# print(Final_data_labels)
#
# Final_data_frame = pd.DataFrame(Final_data_array.transpose(),columns=Final_data_labels)
# print(Final_data_frame)

def calc_rdf(r_cut,bin_size,check_n_atoms,ntypes,Mass,n_part_rdfs,Atom_types,filename):
    ''''''
    n_bins = int(r_cut/bin_size) # may want to write function to ensure that r_cut is a multiple of bin_size
    Rdf_full_sum = np.zeros(n_bins)
    Rdf_part_sum = np.zeros((n_part_rdfs,n_bins))

    Dumps = list(parse_lammps_dumps(filename))
    n_files = len(Dumps)

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
        Radii = (np.arange(n_bins) + 0.5) * bin_size
        data = df.values
        Rdf_full = np.zeros(n_bins)
        Rdf_part = np.zeros((n_part_rdfs, n_bins))
        st = time()
        Rdf_full, Rdf_part = main_loop(data, relation_matrix, n_part_rdfs, Box_lengths, r_cut, bin_size, Rdf_full, Rdf_part)
        print("time:", time() - st)
        print("Finished computing RDF for timestep", Dump.timestep)

        # Normalization Procedure for the full RDF and partical RDFs
        Shell_volume = 4 / 3 * math.pi * bin_size ** 3 * (np.arange(1,n_bins+1) ** 3 - np.arange(n_bins) ** 3)
        Rdf_full = Rdf_full / (n_atoms * rho * Shell_volume)

        Ref_atoms = np.asarray(Atom_types[0]).reshape((n_part_rdfs, 1))
        Ref_atoms_matrix = np.tile(Ref_atoms, n_bins)
        n_atoms_matrix = np.vectorize(atomtypes.get)(Ref_atoms_matrix)

        Rho_n_pairs_matrix = np.tile(rho_n_pairs.reshape((n_part_rdfs,1)), n_bins)

        Shell_volume_matrix = np.tile(Shell_volume, (n_part_rdfs,1))

        # print(Rdf_part)
        # print(n_atoms_matrix)
        # print(rho_n_pairs)
        # print(Rho_n_pairs_matrix)
        # print(Shell_volume_matrix)

        # Rdf_part = Rdf_part / ()
        # Rdf_part_temp = Rdf_part

        Rdf_part = Rdf_part / (n_atoms_matrix * Rho_n_pairs_matrix * Shell_volume_matrix)

        # for index in range(0, n_part_rdfs):
        #     Ref_atom_type = Atom_types[0][index]
        #     Rdf_part[index] = Rdf_part_temp[index] / (atomtypes[Ref_atom_type] * rho_n_pairs * Shell_volume)

        # print(Atom_types)
        # print(rho_n_pairs)
        # print(atomtypes)
        # print(Rdf_part)
        #
        # for kl in range(0, n_part_rdfs):
        #     npp = int(Atom_types[0][kl])
        #     for k in range(0, n_bins):
        #         const = NumConstant * math.pi * pow(bin_size, 3) * (pow((k + 1), 3) - pow(k, 3)) \
        #                 * rho_n_pairs[kl]
        #         if k == n_bins-1:
        #             print(const)
        #         Rdf_part[kl][k] = Rdf_part[kl][k] / (const * atomtypes[npp])
        # for i in range(0, n_bins):
        #     for kl in range(0, n_part_rdfs):
        #         Rdf_part_sum[kl][i] = Rdf_part_sum[kl][i] + Rdf_part[kl][i]

        Rdf_full_sum = Rdf_full_sum + Rdf_full
        Rdf_part_sum = Rdf_part_sum + Rdf_part

        end_traj_loop = timer()
        print('Trajectory loop took: ' + str(end_traj_loop - start_traj_loop) + ' s')

    Rdf_full_sum = Rdf_full_sum / n_files
    Rdf_part_sum = Rdf_part_sum / n_files

    Final_data_array = np.vstack((Radii,Rdf_full_sum,Rdf_part_sum)).transpose()
    Final_data_labels = ['r [Angst]','g_full(r)'] + ['g_' + str(Pair[0]) + '-' + str(Pair[1]) + '(r)' for Pair in relation_matrix]
    Final_data_frame = pd.DataFrame(Final_data_array, columns=Final_data_labels)
    print(Final_data_frame)
    Final_data_frame.to_csv(path_or_buf='rdf.csv',index=False)

    print("Full RDF and partial RDFs are written to rdf.csv file")
    return Final_data_frame

# calc_rdf(r_cut,bin_size,check_n_atoms,ntypes,mass,nrdfs,atom_types,filename)