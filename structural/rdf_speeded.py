
# !/usr/bin/env python3
# -*- coding: utf-8 -*-


# This script calculates full and partial rdf from MD trajectories 

# __author__ = 'Rasha Atwi, Maxim Makeev'
# __version__ = '0.3'
# __email__ = 'rasha.atwi@tufts.edu, maxim.makeev@tufts.edu'
# __date__ = 'Jun 2, 2019'

import os
import sys
import math

import numpy as np
import pandas as pd
from numba import jit, njit
from time import time

@njit(cache=True)
def main_loop(data, relation_matrix, nbin, rcut, ddr, RDF_FULL, RDF_P):

    """
    This function saves the data into a 2d array with column (0) as the id,
    column (2) as type, columns (3, 4, 5) as x, y, and z, column (5) as rsq 
    and column (6) as the bin number. The main for loop is compiled before 
    the code is executed. 
    """

    for i in range(0, data.shape[0] - 1):
        data_head = data[i, :]
        data_i = np.zeros((data.shape[0] - i - 1, data.shape[1] + 2))
        data_i[:, :-2] = data[i + 1:, :]
        data_i[:, 2:5] = data_head[2:] - data_i[:, 2:5]
        dx = data_i[:, 2]
        dy = data_i[:, 3]
        dz = data_i[:, 4]
        cond = (dx > Lx / 2) | (-dx < -Lx / 2)
        dx[cond] = dx[cond] - np.sign(dx[cond]) * Lx
        cond = (dy > Ly / 2) | (-dy < -Ly / 2)
        dy[cond] = dy[cond] - np.sign(dy[cond]) * Ly
        cond = (dz > Lz / 2) | (-dz < -Lz / 2)
        dz[cond] = dz[cond] - np.sign(dz[cond]) * Lz
        rsq = dx ** 2 + dy ** 2 + dz ** 2
        data_i[:, 5] = rsq
        cond = rsq < rcut**2
        data_i = data_i[cond, :]
        data_i[:, 6] = np.sqrt(data_i[:, 5]) / ddr
        for j in data_i[:, 6].astype(np.int64):
            RDF_FULL[j] += 2
        for kl in range(0, n_a_pairs):
            nta1, nta2 = relation_matrix[kl]
            if data_head[1] == nta1:
                v_j = data_i[data_i[:, 1] == nta2]
                for j in  v_j[:, 6].astype(np.int64):
                    RDF_P[kl][j] += 1
            if data_head[1] == nta2:
                v_j = data_i[data_i[:, 1] == nta1]
                for j in  v_j[:, 6].astype(np.int64):
                    RDF_P[kl][j] += 1
    return RDF_FULL, RDF_P

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

# Parameters-loaded from the input file
tmp01 = AI[1].split()
rcut = float(tmp01[0])
tmp02 = AI[3].split()
ddr = float(tmp02[0])
tmp03 = AI[5].split()
NA = int(tmp03[0])
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
tmp10 = AI[21].split()
workdir0 = tmp10[0]
nbin = int(rcut / ddr) + 1
workdir = os.chdir(workdir0)
pathdir = os.getcwd()
print("Working directory:")
print(pathdir)

# For configurations in multiple files, count the number of files
if (input_mode == 'multi'):
    listFiles = []
    temp_array = []
    listoffiles = os.listdir()
    for entry in listoffiles:
        if entry.startswith(filename):
            listFiles.append(entry)
            num_files = len(listFiles)
            print("# of files read:", num_files)

# For configurations in a single file, count the number of frames
num_frame_tot = int(0)
if (input_mode == 'single'):
    filenameONE = filename
    rdffile = open(filenameONE, "r")
    B = rdffile.readlines()
    rdffile.close()
    file_size = len(B)
    print(file_size)
    for ii in range(0, file_size):
        tmp = B[ii].split()
        if (len(tmp) > 1 and tmp[1] == "TIMESTEP"): num_frame_tot += 1
    n_per_frame = int(file_size / num_frame_tot)
    num_files = num_frame_tot

# Define containers to be used to sum configurations
RDF_COOR_SUM = []
RDF_FULL_SUM = []
RDF_P_SUM = []

for i in range(0, nbin):
    RDF_FULL_SUM.append(float(0.0))

for i in range(0, len(nrdf0)):
    RDF_P_SUM.append([0.0] * nbin)

print("num_files=:", num_files)

# Loop over trajectory files
for i_tr in range(0, num_files):
    filename_i = listFiles[i_tr]
    print("The name of the file is:" + filename_i)
    df = pd.read_csv(filename_i, sep=' ', skiprows=9,
                     header=None, index_col=False,
                     names=['id', 'mol', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz',
                            'vx', 'vy', 'vz', 'fx', 'fy', 'fz'])
    df = df[['id', 'type', 'x', 'y', 'z']]
    natoms = df.shape[0]
    with open(filename_i, "r") as file_:
        for i in range(8):
            line = next(file_).split()
            if i == 3:
                NA = int(line[0])
                if (NA != natoms):
                    raise Exception(f"Major consistency check failed: "
                                    "Configuration was not read correctly. NA=: {NA}, "
                                    "natoms=: {natoms}")
                print('Number of atoms=: {0:d}'.format(NA))
            if i == 5: Lx = float(line[1]) - float(line[0])
            if i == 6: Ly = float(line[1]) - float(line[0])
            if i == 7: Lz = float(line[1]) - float(line[0])
    volume = (Lx * Ly * Lz)
    n_a_pairs = len(nrdf0)
    rho_n_pairs = np.zeros(n_a_pairs)
    atomtypes = df.type.value_counts().to_dict()
    setID = set(atomtypes.keys())
    if ntypes != len(setID):
        raise Exception(f"""Consistency check failed:
                Number of atomic types in the config file is 
                different from the corresponding value in input file
                ntypes=: {ntypes}, nset=: {len(setID)}""")
    massT = sum([float(mass[i]) * float(atomtypes[i + 1]) for i in range(ntypes)])
    densT = float((massT / volume) * ConConstant)
    print('{0:s}{1:10.8f}'.format('Average density=:', float(densT)))
    rho = natoms / (Lx * Ly * Lz)
    relation_matrix = np.array([[int(nrdf0[kl]), int(nrdf1[kl])] for kl in range(n_a_pairs)])
    for kk in range(0, n_a_pairs):
        ncurr = int(nrdf1[kk])
        rho_n_pairs[kk] = float(atomtypes[ncurr]) / (Lx * Ly * Lz)
        if rho_n_pairs[kk] < float(1.0e-22):
            raise Exception("Error: Density is zero at kk=:", kk)
    RDF_COOR = []
    for i in range(0, nbin):
        rr = float(ddr * (float(i) + float(0.5)))
        RDF_COOR.append(rr)
    data = df.values
    nbin = np.int64(nbin)
    RDF_FULL = np.zeros(nbin)
    RDF_P = np.zeros((relation_matrix.shape[0], nbin))
    st = time()
    RDF_FULL, RDF_P = main_loop(data, relation_matrix, nbin, rcut, ddr, RDF_FULL, RDF_P)
    print("time:", time() - st)
    print("Finished computing RDF # i_tr", i_tr)

# Normalization Procedure for the full RDF and partical RDFs
    for k in range(0, nbin):
        const = \
        (NumConstant * math.pi) * pow(ddr, 3) * (pow((k + 1), 3) - pow(k, 3)) * rho
        RDF_FULL[k] = RDF_FULL[k] / (const * natoms)
    for kl in range(0, n_a_pairs):
        npp = int(nrdf0[kl])
        for k in range(0, nbin):
            const = NumConstant * math.pi * pow(ddr, 3) * (pow((k + 1), 3) - pow(k, 3)) \
                    * rho_n_pairs[kl]
            RDF_P[kl][k] = RDF_P[kl][k] / (const * atomtypes[npp]) 
    for i in range(0, nbin):
        RDF_FULL_SUM[i] = RDF_FULL_SUM[i] + RDF_FULL[i]
        for kl in range(0, n_a_pairs):
            RDF_P_SUM[kl][i] = RDF_P_SUM[kl][i] + RDF_P[kl][i]
                
for i in range(0, nbin):
    RDF_FULL_SUM[i] = RDF_FULL_SUM[i] / float(num_files)
    for kl in range(0, n_a_pairs):
        RDF_P_SUM[kl][i] = RDF_P_SUM[kl][i] / float(num_files)

outf1 = open("rdf_full.dat", "w+")
for i in range(0, (nbin - 1)):
    outf1.write("%25.20f, %25.20f\n" % (float(RDF_COOR[i]), \
                                        float(RDF_FULL_SUM[i])))
outf1.close()
print("DONE")
for kk in range(0, n_a_pairs):
    p1 = nrdf0[kk]
    p2 = nrdf1[kk]
    filename_p = "rdf_" + str(p1) + str(p2) + ".dat"
    outf = open(filename_p, "w+")
    for i in range(0, nbin - 1):
        outf.write("%25.20f, %25.20f\n" % (float(RDF_COOR[i]), \
                                           float(RDF_P_SUM[kk][i])))
    outf.close()
print("Full RDF and partial RDFs are written to RDF_NM.dat files")   
