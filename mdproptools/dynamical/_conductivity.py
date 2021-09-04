import numpy as np

from mdproptools.common import constants
from mdproptools.common.com_mols import calc_com


def conductivity_loop(dump, num_mols, num_atoms_per_mol, mass, ind, units):
    dump.data = dump.data.sort_values(by=["id"])
    dump.data.reset_index(inplace=True)
    # get dataframe with mol type, mol id, vx, vy, vz, mass, q
    df = calc_com(
        dump,
        num_mols,
        num_atoms_per_mol,
        mass,
        atom_attributes=["vx", "vy", "vz"],
        calc_charge=True,
    )

    # convert to SI units
    df["vx"] = df["vx"] * constants.VELOCITY_CONVERSION[units]
    df["vy"] = df["vy"] * constants.VELOCITY_CONVERSION[units]
    df["vz"] = df["vz"] * constants.VELOCITY_CONVERSION[units]
    df["mass"] = df["mass"] * constants.MASS_CONVERSION[units]
    df["q"] = df["q"] * constants.CHARGE_CONVERSION[units]

    def dot(x):
        return np.dot(x, df.loc[x.index, "q"])

    # for each mol type at a given time, calculate charge flux
    flux = df.groupby("type").agg({"vx": dot, "vy": dot, "vz": dot})
    return (
        dump.timestep * constants.TIME_CONVERSION[units],
        ind,
        flux.reset_index().drop("type", axis=1).T.values,
    )
