import os
import glob
import filecmp
import unittest

from mdproptools.structural.cluster_analysis import get_clusters


class TestClusterAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_files")
        self.dump_files = os.path.join(self.test_dir, "Mg_2TFSI_G1.lammpstrj.*")
        self.elements = ["O", "C", "H", "N", "S", "O", "F", "Mg"]
        self.max_force = 0.75

    def tearDown(self):
        # Clean up generated XYZ files after each test
        xyz_files_test = sorted(glob.glob("*.xyz"))
        for file in xyz_files_test:
            os.remove(file)

    def test_get_clusters_default_type(self):
        num_clusters = self.run_get_clusters(8, False)
        self.assertEqual(num_clusters, 33)

    def test_get_clusters_altered_type(self):
        num_clusters = self.run_get_clusters(32, True)
        self.assertEqual(num_clusters, 33)

    def run_get_clusters(self, atom_type, alter_atom_types):
        num_clusters = get_clusters(
            filename=self.dump_files,
            atom_type=atom_type,
            r_cut=2.3,
            num_mols=[591, 66, 33],
            num_atoms_per_mol=[16, 15, 1],
            full_trajectory=False,
            frame=2,
            elements=self.elements,
            alter_atom_types=alter_atom_types,
            max_force=self.max_force,
            working_dir=None,
        )
        self.compare_generated_xyz_files()
        return num_clusters

    def compare_generated_xyz_files(self):
        xyz_files_test = sorted(glob.glob("*.xyz"))
        xyz_files = sorted(glob.glob(os.path.join(self.test_dir, "*.xyz")))
        self.assertEqual(len(xyz_files_test), len(xyz_files))
        for file1, file2 in zip(xyz_files_test, xyz_files):
            are_same = filecmp.cmp(file1, file2, shallow=False)
            self.assertTrue(are_same, f"{file1} and {file2} are not the same.")

