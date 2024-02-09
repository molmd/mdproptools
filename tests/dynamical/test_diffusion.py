from pathlib import Path

import pytest

import pandas as pd

from mdproptools.dynamical.diffusion import Diffusion


class TestDiffusion:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path):
        self.test_dir = Path(__file__).parent / "test_files"
        self.outputs_dir = Path(__file__).resolve().parents[2] / "data" / "mg_tfsi_dme"
        self.dump_files = "dump.nvt.*.dump"
        self.timestep = 1
        self.units = "real"
        self.diff_dir = tmp_path
        self.d = Diffusion(
            timestep=self.timestep,
            units=self.units,
            outputs_dir=str(self.outputs_dir),
            diff_dir=self.diff_dir,
        )
        yield

    @pytest.mark.parametrize(
        "msd_type, num_mols, num_atoms_per_mol, mass, com_drift, msd_file, all_com_file, int_com_file",
        [
            (
                "com",
                [591, 66, 33],
                [16, 15, 1],
                [
                    16.000,
                    12.010,
                    1.008,
                    14.010,
                    32.060,
                    16.000,
                    12.010,
                    19.000,
                    24.305,
                ],
                True,
                "msd_com.csv",
                "msd_all_com.csv",
                "msd_int_com.csv",
            ),
            (
                "allatom",
                None,
                None,
                None,
                False,
                "msd_allatom.csv",
                "msd_all_allatom.csv",
                "msd_int_allatom.csv",
            ),
        ],
    )
    def test_get_msd_from_dump(
        self,
        benchmark,
        msd_type,
        num_mols,
        num_atoms_per_mol,
        mass,
        com_drift,
        msd_file,
        all_com_file,
        int_com_file,
    ):
        msd_test, msd_all_test, msd_int_test = benchmark(
            self.d.get_msd_from_dump,
            filename=self.dump_files,
            msd_type=msd_type,
            num_mols=num_mols,
            num_atoms_per_mol=num_atoms_per_mol,
            mass=mass,
            com_drift=com_drift,
            avg_interval=True,
            tao_coeff=4,
        )

        msd = pd.read_csv(str(self.test_dir / msd_file))
        pd.testing.assert_frame_equal(msd_test, msd)

        msd_all = pd.read_csv(str(self.test_dir / all_com_file))
        pd.testing.assert_frame_equal(msd_all_test, msd_all)

        msd_int = pd.read_csv(str(self.test_dir / int_com_file))
        pd.testing.assert_frame_equal(msd_int_test, msd_int)

    def test_get_msd_from_log(self, benchmark):
        msd_log_test = benchmark(self.d.get_msd_from_log, log_pattern="log.mixture_nvt")

        msd_log = pd.read_csv(str(self.test_dir / "msd_log.csv"))
        pd.testing.assert_frame_equal(msd_log_test, msd_log)

    def test_calc_diff(self, benchmark):
        msd_df = pd.read_csv(str(self.test_dir / "msd_com.csv"))

        diffusion_test = benchmark(
            self.d.calc_diff,
            msd=msd_df,
            initial_time=None,
            final_time=None,
            dimension=3,
            diff_names=["dme", "tfsi", "mg"],
            save=True,
            plot=True,
        )

        diffusion = pd.read_csv(str(self.test_dir / "diffusion.csv"), index_col=0)
        pd.testing.assert_frame_equal(diffusion_test, diffusion)

        diff_files = [
            self.diff_dir / "diff_dme.txt",
            self.diff_dir / "diff_tfsi.txt",
            self.diff_dir / "diff_mg.txt",
        ]

        png_files = [self.diff_dir / "msd.png", self.diff_dir / "msd_log.png"]

        for file_path in diff_files + png_files:
            assert file_path.exists(), f"File {file_path} was not created."

    def test_get_diff_dist(self, benchmark):
        msd_int = pd.read_csv(str(self.test_dir / "msd_int_com.csv"))

        diff_dist_test = benchmark(
            self.d.get_diff_dist,
            msd_int=msd_int,
            dump_freq=50000,
            dimension=3,
            tao_coeff=4,
            plot=True,
            diff_names=["dme", "tfsi", "mg"],
        )

        diff_dist = pd.read_csv(str(self.test_dir / "diff_dist.csv"))
        pd.testing.assert_frame_equal(diff_dist_test, diff_dist)

        png_file = self.diff_dir / "diff_dist.png"
        assert png_file.exists(), f"File {png_file} was not created."
