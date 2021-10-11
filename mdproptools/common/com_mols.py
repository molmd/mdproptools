import pandas as pd
import numpy as np


def calc_com(
    dump,
    num_mols,
    num_atoms_per_mol,
    mass=None,
    atom_attributes=["xu", "yu", "zu"],
    calc_charge=False,
):
    """
    Calculates the center of mass of each molecule.
    Args:
        dump (LammpsDump): pymatgen LAMMPS dump object; should be sorted by id
        num_mols (list): Number of each molecule type in the system
        num_atoms_per_mol (list): Number of atoms in each molecule type; order is
            similar to that in num_mols
        mass (list, optional): Atomic masses; order is similar to that of the atom
            types in the LAMMPS data file; Not required if the mass is available in the
            dump
        atom_attributes (list): Atom attributes used in the center of mass calculations;
            Defaults to unwrapped coordinates ["xu", "yu", "zu"]
        calc_charge (bool): Whether to calculate the charge of each molecule;
            Defaults to False

    Returns:
        pd.Dataframe with mol types, mol ids, com attributes, molecule mass, and q (optional)
    """
    mol_types = [
        mol_type + 1
        for mol_type, number_of_mols in enumerate(num_mols)
        for mol_id in range(number_of_mols)
        for atom_id in range(num_atoms_per_mol[mol_type])
    ]
    mol_ids = [
        mol_id + 1
        for mol_type, number_of_mols in enumerate(num_mols)
        for mol_id in range(number_of_mols)
        for atom_id in range(num_atoms_per_mol[mol_type])
    ]
    if calc_charge:
        attributes = atom_attributes + ["q"]
    else:
        attributes = atom_attributes
    if not mass:
        assert "mass" in dump.data.columns, "Missing atom masses in dump file."
        columns = ["id", "type", "mass"] + attributes
        df = dump.data[columns].copy()
    else:
        columns = ["id", "type"] + attributes
        df = dump.data[columns].copy()
        df["mass"] = df.apply(lambda x: mass[int(x.type - 1)], axis=1)
    df["mol_type"] = np.array(mol_types)
    df["mol_id"] = np.array(mol_ids)
    df = df.drop(["type", "id"], axis=1)
    df[atom_attributes] = df[atom_attributes].multiply(df["mass"], axis=0)
    mol_df = df.groupby(["mol_type", "mol_id"]).sum()
    mol_df[atom_attributes] = mol_df[atom_attributes].divide(mol_df["mass"], axis=0)
    mol_df.index = mol_df.index.rename("type", level="mol_type")
    return mol_df
