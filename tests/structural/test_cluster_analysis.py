import glob, filecmp

from pathlib import Path

import pytest

import pandas as pd

from pymatgen.core.structure import Molecule

from mdproptools.structural.cluster_analysis import (
    get_clusters,
    get_unique_configurations,
)


class TestClusterAnalysis:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path):
        self.test_dir = Path(__file__).parent / "test_files"
        self.data_dir = Path(__file__).resolve().parents[2] / "data" / "mg_tfsi_dme"
        self.dump_files = str(self.data_dir / "dump.nvt.*.dump")
        self.elements = ["O", "C", "H", "N", "S", "O", "C", "F", "Mg"]
        self.r_cut = 2.3
        self.max_force = 0.75
        self.working_dir = tmp_path
        yield

    @pytest.fixture
    def load_molecules(self):
        dme = Molecule.from_file(str(self.data_dir / "dme.pdb"))
        tfsi = Molecule.from_file(str(self.data_dir / "tfsi.pdb"))
        mg = Molecule.from_file(str(self.data_dir / "mg.pdb"))
        return [dme, tfsi, mg]

    def compare_generated_xyz_files(self, pattern):
        xyz_files_test = sorted(glob.glob(str(self.working_dir / pattern)))
        xyz_files = sorted(glob.glob(str(self.test_dir / pattern)))
        assert len(xyz_files_test) == len(xyz_files), "Number of XYZ files mismatch."
        for file1, file2 in zip(xyz_files_test, xyz_files):
            are_same = filecmp.cmp(file1, file2, shallow=False)
            assert are_same, f"{file1} and {file2} are not the same."

    def test_get_clusters(self, benchmark):
        num_clusters = benchmark(
            get_clusters,
            filename=self.dump_files,
            atom_type=9,
            r_cut=self.r_cut,
            num_mols=[591, 66, 33],
            num_atoms_per_mol=[16, 15, 1],
            full_trajectory=False,
            frame=50,
            elements=self.elements,
            alter_atom_types=False,
            max_force=self.max_force,
            working_dir=self.working_dir,
        )
        assert num_clusters == 33
        self.compare_generated_xyz_files("Cluster_*.xyz")

    def test_get_unique_configurations(self, benchmark, load_molecules):
        get_clusters(
            filename=self.dump_files,
            atom_type=32,
            r_cut=self.r_cut,
            num_mols=[591, 66, 33],
            num_atoms_per_mol=[16, 15, 1],
            full_trajectory=False,
            frame=50,
            elements=self.elements,
            alter_atom_types=True,
            max_force=self.max_force,
            working_dir=self.working_dir,
        )

        clusters_df_test, conf_df_test = benchmark(
            get_unique_configurations,
            cluster_pattern="Cluster_*.xyz",
            r_cut=self.r_cut,
            molecules=load_molecules,
            mol_num=2,
            type_coord_atoms=["O", "N", "Mg"],
            working_dir=self.working_dir,
            find_top=True,
            perc=None,
            cum_perc=100,
            mol_names=["dme", "tfsi", "mg"],
            zip=False,
        )
        self.compare_generated_xyz_files("conf_*.xyz")

        cluster_df = pd.read_csv(str(self.test_dir / "clusters.csv"))
        cluster_df.fillna("", inplace=True)
        pd.testing.assert_frame_equal(clusters_df_test, cluster_df, check_dtype=False)

        top_conf_df = pd.read_csv(str(self.test_dir / "top_conf.csv"))
        top_conf_df_test = pd.read_csv(str(self.working_dir / "top_conf.csv"))
        assert len(top_conf_df_test) == 5
        pd.testing.assert_frame_equal(top_conf_df_test, top_conf_df)
