#!/usr/bin/env python

"""
Contains scientific constants and conversion factors into SI units for LAMMPS units.
The constants are from:
BIPM. Le Système international d’unités / The International System of Units (‘The SI Brochure’).
    Bureau international des poids et mesures, ninth edition, 2019. URL
    http://www.bipm.org/en/si/si_brochure/, ISBN 978-92-822-2272-0.
"2018 CODATA Value: Bohr radius". The NIST Reference on Constants, Units, and Uncertainty. NIST.
    20 May 2019.
"2018 CODATA Value: Hartree energy". The NIST Reference on Constants, Units, and Uncertainty. NIST.
    20 May 2019.
"""

__author__ = "Matthew Bliss"
__maintainer__ = "Matthew Bliss"
__email__ = "matthew.bliss@stonybrook.edu"
__status__ = "Development"
__date__ = "Jun 2020"
__version__ = "0.0.1"

BOLTZMANN = 1.380649 * 10 ** -23  # J/K

ELEMENTARY_CHARGE = 1.602176634 * 10 ** -19  # C

AVOGADRO = 6.02214076 * 10 ** 23  # 1/mol

LIGHT_SPEED = 299792458  # m/s

BOHR_RADIUS = 5.29177210903 * 10 ** -11  # m

CAL_TO_J = 4.184  # J/cal

HA_TO_J = 4.3597447222071 * 10 ** -18


SUPPORTED_UNITS = ["real", "metal", "si", "cgs", "electron", "micro", "nano"]

MASS_CONVERSION = {
    "real": 10 ** -3 / AVOGADRO,  # from grams/mole to kg
    "metal": 10 ** -3 / AVOGADRO,  # from grams/mole to kg
    "si": 1,  # from kilograms to kg
    "cgs": 10 ** -3,  # from grams to kg
    "electron": 10 ** -3 / AVOGADRO,  # from grams/mole to kg
    "micro": 10 ** -3 * 10 ** -12,  # from picograms to kg
    "nano": 10 ** -3 * 10 ** -18,
}  # from attograms to kg

DISTANCE_CONVERSION = {
    "real": 10 ** -10,  # from Angstroms to m
    "metal": 10 ** -10,  # from Angstroms to m
    "si": 1,  # from meters to m
    "cgs": 10 ** -2,  # from centimeters to m
    "electron": BOHR_RADIUS,  # from Bohr radius to m
    "micro": 10 ** -6,  # from micrometers to m
    "nano": 10 ** -9,
}  # from nanometers to m

TIME_CONVERSION = {
    "real": 10 ** -15,  # from femtoseconds to s
    "metal": 10 ** -12,  # from picoseconds to s
    "si": 1,  # from seconds to s
    "cgs": 1,  # from seconds to s
    "electron": 10 ** -15,  # from femtoseconds to s
    "micro": 10 ** -6,  # from microseconds to s
    "nano": 10 ** -9,
}  # from nanoseconds to s

ENERGY_CONVERSION = {
    "real": 10 ** 3 * CAL_TO_J / AVOGADRO,  # kcal/mol to J
    "metal": ELEMENTARY_CHARGE,  # eV to J
    "si": 1,  # Joules to J
    "cgs": 10 ** -7,  # erg to J
    "electron": HA_TO_J,  # Hartree to J
    "micro": MASS_CONVERSION["micro"],  # pg*mum^2/mus^2 to J
    "nano": MASS_CONVERSION["nano"],
}  # ag*nm^2/ns^2 to J

VELOCITY_CONVERSION = {
    "real": DISTANCE_CONVERSION["real"] / TIME_CONVERSION["real"],
    "metal": DISTANCE_CONVERSION["metal"] / TIME_CONVERSION["metal"],
    "si": 1,
    "cgs": DISTANCE_CONVERSION["cgs"] / TIME_CONVERSION["cgs"],
    "electron": DISTANCE_CONVERSION["electron"] / (1.03275 * 10 ** -15),
    "micro": DISTANCE_CONVERSION["micro"] / TIME_CONVERSION["micro"],
    "nano": DISTANCE_CONVERSION["nano"] / TIME_CONVERSION["nano"],
}

FORCE_CONVERSION = {
    "real": ENERGY_CONVERSION["real"] / DISTANCE_CONVERSION["real"],
    "metal": ENERGY_CONVERSION["metal"] / DISTANCE_CONVERSION["metal"],
    "si": 1,
    "cgs": ENERGY_CONVERSION["cgs"] / DISTANCE_CONVERSION["cgs"],
    "electron": ENERGY_CONVERSION["electron"] / DISTANCE_CONVERSION["electron"],
    "micro": ENERGY_CONVERSION["micro"] / DISTANCE_CONVERSION["micro"],
    "nano": ENERGY_CONVERSION["nano"] / DISTANCE_CONVERSION["nano"],
}

TORQUE_CONVERSION = ENERGY_CONVERSION

TEMPERATURE_CONVERSION = {
    "real": 1,
    "metal": 1,
    "si": 1,
    "cgs": 1,
    "electron": 1,
    "micro": 1,
    "nano": 1,
}

PRESSURE_CONVERSION = {
    "real": 101325,  # atm to Pa
    "metal": 10 ** 5,  # bar to Pa
    "si": 1,  # Pa to Pa
    "cgs": 10 ** -6 * 10 ** 5,  # dyne/cm^2 to Pa
    "electron": 1,  # Pa
    "micro": ENERGY_CONVERSION["micro"] / DISTANCE_CONVERSION["micro"] ** 3,
    "nano": ENERGY_CONVERSION["nano"] / DISTANCE_CONVERSION["nano"] ** 3,
}

VISCOSITY_CONVERSION = {
    "real": 0.1,  # P to Pa*s
    "metal": 0.1,  # P to Pa*s
    "si": 1,  # Pa*s to Pa*s
    "cgs": 0.1,  # P to Pa*s
    "electron": 1,  # Pa*s to Pa*s
    "micro": PRESSURE_CONVERSION["micro"] * TIME_CONVERSION["micro"],
    "nano": PRESSURE_CONVERSION["nano"] * TIME_CONVERSION["nano"],
}

CHARGE_CONVERSION = {
    "real": ELEMENTARY_CHARGE,  # e to C
    "metal": ELEMENTARY_CHARGE,  # e to C
    "si": 1,  # C to C
    "cgs": 1 / 10 / LIGHT_SPEED,  # esu to C
    "electron": ELEMENTARY_CHARGE,  # e to C
    "micro": 10 ** -12,  # pC to C
    "nano": ELEMENTARY_CHARGE,
}  # e to C

DIPOLE_CONVERSION = {
    "real": CHARGE_CONVERSION["real"] * DISTANCE_CONVERSION["real"],
    "metal": CHARGE_CONVERSION["metal"] * DISTANCE_CONVERSION["metal"],
    "si": 1,
    "cgs": CHARGE_CONVERSION["cgs"] * DISTANCE_CONVERSION["cgs"],
    "electron": 10 ** -21 / LIGHT_SPEED,
    "micro": CHARGE_CONVERSION["micro"] * DISTANCE_CONVERSION["micro"],
    "nano": CHARGE_CONVERSION["nano"] * DISTANCE_CONVERSION["nano"],
}

ELECTRIC_FIELD_CONVERSION = {
    "real": 1 / DISTANCE_CONVERSION["real"],
    "metal": 1 / DISTANCE_CONVERSION["metal"],
    "si": 1,
    "cgs": FORCE_CONVERSION["cgs"] / CHARGE_CONVERSION["cgs"],
    "electron": 1 / 10 ** -2,
    "micro": 1 / DISTANCE_CONVERSION["micro"],
    "nano": 1 / DISTANCE_CONVERSION["nano"],
}

DENSITY_3D_CONVERSION = {
    "real": MASS_CONVERSION["cgs"] / DISTANCE_CONVERSION["cgs"] ** 3,
    "metal": MASS_CONVERSION["cgs"] / DISTANCE_CONVERSION["cgs"] ** 3,
    "si": 1,
    "cgs": MASS_CONVERSION["cgs"] / DISTANCE_CONVERSION["cgs"] ** 3,
    "micro": MASS_CONVERSION["micro"] / DISTANCE_CONVERSION["micro"] ** 3,
    "nano": MASS_CONVERSION["nano"] / DISTANCE_CONVERSION["nano"] ** 3,
}
