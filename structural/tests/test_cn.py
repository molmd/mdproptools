# coding: utf-8
import os
import unittest
import pandas as pd

from analysis.lammps.structural.rdf_cn import calc_atomic_cn, calc_molecular_cn

__author__ = "Rasha Atwi"
__version__ = "0.1.0"
__email__ = "rasha.atwi@tufts.edu"
__date__ = "Dec 10, 2019"

test_dir = os.path.join(os.path.dirname(__file__), '..', 'test_files')
dumps_dir = os.path.join(test_dir, "Mg_2TFSI_G1.lammpstrj.*")


class CNTest(unittest.TestCase):

    def test_atomic_cn(self):
        cn_default_ids_test = calc_atomic_cn([2.325, 4.375, 2.375, 13.0], 0.05,
                                             8, [16, 12.01, 1.008, 14.01, 32.06,
                                                 16, 19, 24.305], [[8, 8, 8, 8],
                                                                   [1, 4, 6, 8]],
                                             dumps_dir, save_mode=False)

        cn_default_ids = pd.read_csv(os.path.join(test_dir, "cn_default_ids.csv"),
                                     delimiter=",")

        cn_altered_ids_test = calc_atomic_cn([4.375, 13.0], 0.05, 8,
                                             [16, 12.01, 1.008, 14.01, 32.06,
                                              16, 19, 24.305], [[32, 32],
                                                                [17, 32]],
                                             dumps_dir, num_mols=[591, 66, 33],
                                             num_atoms_per_mol=[16, 15, 1],
                                             save_mode=False)

        cn_altered_ids = pd.read_csv(os.path.join(test_dir, "cn_altered_ids.csv"),
                                     delimiter=",")

        self.assertTrue(cn_default_ids.round(10).equals(cn_default_ids_test.
                                                        round(10)))
        self.assertTrue(cn_altered_ids.round(10).equals(cn_altered_ids_test.
                                                        round(10)))

    def test_molecular_cn(self):
        cn_mol_test = calc_molecular_cn([2.325, 3.775, 4.375], 0.05, 8,
                                        [16, 12.01, 1.008, 14.01, 32.06, 16, 19,
                                         24.305], [[8, 8, 4], [1, 2, 3]],
                                        dumps_dir, num_mols=[591, 66, 33],
                                        num_atoms_per_mol=[16, 15, 1],
                                        save_mode=False)
        print(cn_mol_test)
        cn_mol = pd.read_csv(os.path.join(test_dir, "cn_mol.csv"),
                             delimiter=",")

        self.assertTrue(cn_mol.round(10).equals(cn_mol_test.round(10)))


if __name__ == '__main__':
    unittest.main()
