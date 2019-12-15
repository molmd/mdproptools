# !/usr/bin/env python3
# -*- coding: utf-8 -*-


# This script calculates full and partial rdfs via the functions in the rdf_speeded.py module

# __author__ = 'Rasha Atwi, Maxim Makeev, Matthew Bliss'
# __version__ = '0.1'
# __email__ = 'rasha.atwi@tufts.edu, maxim.makeev@tufts.edu, matthew.bliss@tufts.edu'
# __date__ = 'Oct 9, 2019'

import os
import glob
import structural.rdf_speeded as rdf

# # Defined inputs
r_cut = 20
bin_size = 0.2
check_n_atoms = 35253
ntypes = 12
mass = [14.010, 12.010, 12.010, 12.010, 1.008, 16.000, 32.060, 16.000, 1.008, 16.000, 1.008, 22.990]
nrdfs = 7
atom_types = [[12, 12, 12, 12, 12, 6, 1],
              [ 1,  6,  8, 10, 12, 6, 1]]
filename = '../dump.0.5M_dhps_2.5M_na_1M_oh.nvt_298.15K.*'
# filename = 'dump.0.5M_dhps_2.5M_na_1M_oh.nvt_298.15K.0.lammpstrj'
# os.chdir('./')

# # # For troubleshooting problems with parse_lammps_dumps()
# pathdir = os.getcwd()
# print(pathdir)
# # path_to_file = os.path.join(pathdir,'test_files',filename)
# # print(path_to_file)
#
# files = glob.glob(filename)
# print(files)

rdf.calc_rdf(r_cut,bin_size,check_n_atoms,ntypes,mass,nrdfs,atom_types,filename)

