from pathlib import Path

import pytest

import pandas as pd

from pymatgen.io.lammps.outputs import parse_lammps_log

from mdproptools.utilities.fluctuations import plot_fluctuations


class TestFluctuations:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path):
        self.data_dir = Path(__file__).resolve().parents[2] / "data" / "mg_tfsi_dme"
        self.test_dir = Path(__file__).parent / "test_files"
        self.working_dir = tmp_path
        yield

    def run_fluctuations(self):
        log_file = self.data_dir / "log.mixture_npt"
        df = pd.read_csv(self.test_dir / "prop_stats.csv")
        log = parse_lammps_log(str(log_file))
        props = {
            "Temp": ["Temperature (K)", "temperature_fluctuations"],
            "Press": ["Pressure (atm)", "pressure_fluctuations"],
            "TotEng": ["Total Energy (Kcal/mole)", "total_energy_fluctuations"],
            "PotEng": ["Potential Energy (Kcal/mole)", "potential_energy_fluctuations"],
            "KinEng": ["Kinetic Energy (Kcal/mole)", "kinetic_energy_fluctuations"],
            "Volume": ["Volume ($\AA^3$)", "volume_fluctuations"],
            "Density": ["Density (g/$cm^3$)", "density_fluctuations"],
            "Lx": ["Length ($\AA$)", "length_fluctuations"],
            "E_bond": ["Bond Energy (Kcal/mole)", "bond_energy_fluctuations"],
            "E_angle": ["Angle Energy (Kcal/mole)", "angle_energy_fluctuations"],
            "E_dihed": ["Dihedral Energy (Kcal/mole)", "dihedral_energy_fluctuations"],
            "E_pair": ["Pairwise Energy (Kcal/mole)", "pairwise_energy_fluctuations"],
            "E_vdwl": ["van der Waals Energy (Kcal/mole)", "vdwl_energy_fluctuations"],
            "E_coul": ["Coulombic Energy (Kcal/mole)", "coulombic_energy_fluctuations"],
            "E_long": [
                "Long-range Kspace Energy (Kcal/mole)",
                "longrange_energy_fluctuations",
            ],
            "Fnorm": ["Force (Kcal/mole.$\AA$)", "force_fluctuations"],
        }

        prop_stats = {}
        for prop, (title, filename) in props.items():
            prop_mean, prop_std = plot_fluctuations(
                log[0],
                prop,
                title,
                filename,
                timestep=1,
                units="real",
                working_dir=self.working_dir,
            )
            prop_stats[prop] = (prop_mean, prop_std)
        df_test = pd.DataFrame(prop_stats).T.reset_index()
        df_test.columns = ["prop", "mean", "std"]
        pd.testing.assert_frame_equal(df_test, df)

    def test_fluctuations(self, benchmark):
        benchmark(self.run_fluctuations)
