# !/usr/bin/env python3
# -*- coding: utf-8 -*-


# This script calculates full and partial rdfs via the functions in the rdf_speeded.py module

# __author__ = 'Rasha Atwi, Maxim Makeev, Matthew Bliss'
# __version__ = '0.1'
# __email__ = 'rasha.atwi@tufts.edu, maxim.makeev@tufts.edu, matthew.bliss@tufts.edu'
# __date__ = 'Oct 9, 2019'

import os
import glob
import structural.rdf_new_atomtypes_functions as rdf

# # Defined inputs
r_cut = 20
bin_size = 0.2
check_n_atoms = 35253
ntypes = 12
mass = [14.010, 12.010, 12.010, 12.010, 1.008, 16.000, 32.060, 16.000, 1.008, 16.000, 1.008, 22.990]
nrdfs = 10
atom_types = [[31, 31, 31, 31, 31, 31, 31, 31, 31, 31],
              [ 1,  2, 16, 20, 21, 24, 25, 26, 29, 31]]
n_mols_of_each_type = [105, 10561, 210, 525]
n_atoms_in_each_mol = [25, 3, 2, 1]

filename = '../dump.0.5M_dhps_2.5M_na_1M_oh.nvt_298.15K.*'
rdf_data = rdf.calc_rdf(r_cut, bin_size, check_n_atoms, ntypes, mass, nrdfs, atom_types,filename, nmols=n_mols_of_each_type, natoms_per_mol=n_atoms_in_each_mol)

print(rdf_data)