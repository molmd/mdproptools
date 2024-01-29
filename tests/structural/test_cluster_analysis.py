import os, glob, filecmp, unittest

from pathlib import Path

import pandas as pd

from pymatgen.core.structure import Molecule

from mdproptools.structural.cluster_analysis import (
    get_clusters,
    get_unique_configurations,
)


class TestClusterAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).parent / "test_files"
        self.data_dir = Path(__file__).resolve().parents[2] / "data" / "structural"
        self.dump_files = str(self.data_dir / "Mg_2TFSI_G1.lammpstrj.*")
        self.elements = ["O", "C", "H", "N", "S", "O", "F", "Mg"]
        self.r_cut = 2.3
        self.max_force = 0.75

    def tearDown(self):
        # Clean up generated XYZ files after each test
        xyz_files_test = sorted(glob.glob("*.xyz"))
        for file in xyz_files_test:
            os.remove(file)

        # Clean up generated CSV files after each test
        csv_files_test = sorted(glob.glob("*.csv"))
        for file in csv_files_test:
            os.remove(file)

    def test_get_clusters(self):
        num_clusters = self.run_get_clusters(8, False)
        self.assertEqual(num_clusters, 33)

    def test_get_unique_configurations(self):
        num_clusters = self.run_get_clusters(32, True)
        self.assertEqual(num_clusters, 33)

        dme = Molecule.from_file(str(self.data_dir / "dme.pdb"))
        tfsi = Molecule.from_file(str(self.data_dir / "tfsi.pdb"))
        mg = Molecule.from_file(str(self.data_dir / "mg.pdb"))

        clusters_df_test, conf_df_test = get_unique_configurations(
            cluster_pattern="Cluster_*.xyz",
            r_cut=self.r_cut,
            molecules=[dme, tfsi, mg],
            type_coord_atoms=["O", "N", "Mg"],
            working_dir=None,
            find_top=True,
            perc=None,
            cum_perc=80,
            mol_names=["dme", "tfsi", "mg"],
            zip=False,
        )
        self.compare_generated_xyz_files("conf_*.xyz")

        cluster_df = pd.read_csv(str(self.test_dir / "clusters.csv"))
        cluster_df.fillna("", inplace=True)
        pd.testing.assert_frame_equal(clusters_df_test, cluster_df, check_dtype=False)

        top_conf_df = pd.read_csv(str(self.test_dir / "top_conf.csv"))
        top_conf_df_test = pd.read_csv("top_conf.csv")
        self.assertEqual(len(top_conf_df_test), 3)
        pd.testing.assert_frame_equal(top_conf_df_test, top_conf_df)

    def run_get_clusters(self, atom_type, alter_atom_types):
        num_clusters = get_clusters(
            filename=self.dump_files,
            atom_type=atom_type,
            r_cut=self.r_cut,
            num_mols=[591, 66, 33],
            num_atoms_per_mol=[16, 15, 1],
            full_trajectory=False,
            frame=2,
            elements=self.elements,
            alter_atom_types=alter_atom_types,
            max_force=self.max_force,
            working_dir=None,
        )
        self.compare_generated_xyz_files("Cluster_*.xyz")
        return num_clusters

    def compare_generated_xyz_files(self, pattern):
        xyz_files_test = sorted(glob.glob(pattern))
        xyz_files = sorted(glob.glob(str(self.test_dir / pattern)))
        self.assertEqual(len(xyz_files_test), len(xyz_files))
        for file1, file2 in zip(xyz_files_test, xyz_files):
            are_same = filecmp.cmp(file1, file2, shallow=False)
            self.assertTrue(are_same, f"{file1} and {file2} are not the same.")
