import pandas as pd


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
    mol_types = []
    mol_ids = []

    for mol_type, number_of_mols in enumerate(num_mols):
        for mol_id in range(number_of_mols):
            for atom_id in range(num_atoms_per_mol[mol_type]):
                mol_types.append(mol_type + 1)
                mol_ids.append(mol_id + 1)
    if calc_charge:
        attributes = atom_attributes + ["q"]
    else:
        attributes = atom_attributes
    if not mass:
        assert "mass" in dump.data.columns, "Missing atom masses in dump file."
        columns = ["id", "type", "mass"] + attributes
        df = pd.DataFrame(dump.data[columns], columns=columns)
    else:
        columns = ["id", "type"] + attributes
        df = pd.DataFrame(dump.data[columns], columns=columns)
        df["mass"] = df.apply(lambda x: mass[int(x.type - 1)], axis=1)

    df["mol_type"] = mol_types
    df["mol_id"] = mol_ids
    df = df.drop(["type", "id"], axis=1).set_index(["mol_type", "mol_id"])
    mol_df = df.groupby(["mol_type", "mol_id"]).apply(
        lambda x: pd.Series(
            x["mass"].values @ x[atom_attributes].values / x.mass.sum(),
            index=atom_attributes,
        )
    )
    mol_df["mass"] = df.groupby(["mol_type", "mol_id"])["mass"].sum()
    if calc_charge:
        mol_df["q"] = df.groupby(["mol_type", "mol_id"])["q"].sum()
    return mol_df.reset_index().rename(columns={"mol_type": "type"})
