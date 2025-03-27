#!/usr/bin/env python

"""
Extracts clusters within a cutoff distance from an atom from LAMMPS trajectory files.
"""

import os, glob, ntpath, shutil, warnings

from collections import Counter

import numpy as np
import pandas as pd

from tqdm import tqdm

from pymatgen.core.structure import Molecule
from pymatgen.io.lammps.outputs import parse_lammps_dumps

from mdproptools.structural.rdf_cn import _calc_rsq, _calc_atom_type

__author__ = "Rasha Atwi, Maxim Makeev"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "Apr 2020"
__version__ = "0.0.6"

FORCE_CONSTANT = 0.043363 / 16.0


def _remove_boundary_effects(data_head, mol_data, lx, ly, lz, num_of_ids):
    data_i = np.zeros((mol_data.shape[0], mol_data.shape[1]))
    data_i[:, : num_of_ids + 3] = mol_data.copy()
    dxyz = mol_data[:, num_of_ids:] - data_head[num_of_ids:]
    dx = dxyz[:, 0]
    dy = dxyz[:, 1]
    dz = dxyz[:, 2]
    cond = (dx > lx / 2) | (dx < -lx / 2)
    data_i[cond, num_of_ids] = data_i[cond, num_of_ids] - np.sign(dx[cond]) * lx
    cond = (dy > ly / 2) | (dy < -ly / 2)
    data_i[cond, num_of_ids + 1] = data_i[cond, num_of_ids + 1] - np.sign(dy[cond]) * ly
    cond = (dz > lz / 2) | (dz < -lz / 2)
    data_i[cond, num_of_ids + 2] = data_i[cond, num_of_ids + 2] - np.sign(dz[cond]) * lz
    return data_i


def get_clusters(
    filename,
    atom_type,
    r_cut,
    num_mols,
    num_atoms_per_mol,
    full_trajectory=False,
    frame=None,
    elements=None,
    alter_atom_types=False,
    max_force=0.75,
    working_dir=None,
):
    """
    Extracts clusters within a cutoff distance from an atom from LAMMPS trajectory files
    and saves them as *.xyz files in the working directory. The LAMMPS dump files
    should contain the following attributes: id, type, x, y, z, fx, fy, fz. Additionally,
    the elements of the atoms in the system should be provided if they are not in the
    dump files.

    Args:
        filename (str or file handle): the name of the LAMMPS dump file; can be
            the entire name for a single file or a file pattern with the
            wildcard character ('*') for multiple dumps; should contain the path to the
            dumps if they are not placed in the same directory
        atom_type (int): the type number of the atom (as it appears in the LAMMPS data
            file) around which the clusters are extracted
        r_cut (float): the cutoff distance for the clusters in Angstroms
        num_mols (list of int): the number of molecules of each type in the system
        num_atoms_per_mol (list of int): the number of atoms in each molecule type
        full_trajectory (bool, optional): if True, all frames in the trajectory are
            processed
        frame (int, optional): the frame number to be processed if full_trajectory is
            set to False
        elements (list of str, optional): name of the atom elements in the system;
            required if the elements are not in the LAMMPS dump file
        alter_atom_types (bool, optional): if True, new atom types will be calculated to
            distinguish between two similar atoms in the same molecule or a different
            molecule; defaults to False if default LAMMPS atom types are to be used; if
            True, the atom_type argument should match the new atom types and not the
            default LAMMPS atom types
        max_force (float, optional): the maximum force on the atom in eV/Angstroms
        working_dir (str, optional): path to the working directory where the cluster
            files will be saved; defaults to the current working directory

    Returns:
        int: the number of clusters written to *.xyz files
    """
    # Map the elements to the atom types
    if elements:
        elements = {i + 1: j for i, j in enumerate(elements)}
    if not working_dir:
        working_dir = os.getcwd()
    dumps = list(parse_lammps_dumps(filename))
    if full_trajectory:
        dumps = dumps
    else:
        dumps = [dumps[frame]]

    # Iterate over the frames in the trajectory and extract the clusters
    cluster_count = 0
    for index, dump in enumerate(tqdm(dumps, desc="Processing dump files")):
        # Calculate the box dimensions and sort the data by atom id
        lx = dump.box.bounds[0][1] - dump.box.bounds[0][0]
        ly = dump.box.bounds[1][1] - dump.box.bounds[1][0]
        lz = dump.box.bounds[2][1] - dump.box.bounds[2][0]
        df = dump.data.sort_values(by=["id"])

        # Calculate the molecule type and id for each atom
        mol_types = []
        mol_ids = []
        for mol_type, number_of_mols in enumerate(num_mols):
            for mol_id in range(number_of_mols):
                for atom_id in range(num_atoms_per_mol[mol_type]):
                    mol_types.append(mol_type + 1)
                    mol_ids.append(mol_id + 1)
        df["mol_type"] = mol_types
        df["mol_id"] = mol_ids

        if "element" not in df.columns and not elements:
            raise ValueError(
                "The elements of the atoms in the system should be provided if they "
                "are not in the dump files."
            )

        if elements:
            df["element"] = df["type"].map(elements)

        # Calculate the new atom types if requested by the user
        if alter_atom_types:
            data = _calc_atom_type(df.values, num_mols, num_atoms_per_mol)
            df["type"] = data[:, 0]

        # Iterate over the atoms of the specified type and extract the clusters
        counter = 0
        atoms = df[df["type"] == atom_type]["id"]
        for i in atoms:
            counter += 1
            data_head_i = df[df["id"] == i][
                ["mol_type", "mol_id", "x", "y", "z"]
            ].values[0]

            # Calculate the squared distance between the atom and all other atoms
            data_all, rsq = _calc_rsq(
                data_head_i,
                df.loc[:, ["mol_type", "mol_id", "x", "y", "z"]].values,
                lx,
                ly,
                lz,
                2,
            )

            # Extract the atoms within the cutoff distance from the atom
            cond = rsq < r_cut**2
            data_all = data_all[cond, :]
            ids = pd.DataFrame(
                np.unique(data_all[:, [0, 1]], axis=0), columns=["mol_type", "mol_id"]
            )
            neighbor_df = ids.merge(df, on=["mol_type", "mol_id"])

            # Calculate the minimum force on the atoms within each molecule in the
            # neighbor dataframe
            min_force = (
                neighbor_df.groupby(["mol_type", "mol_id"])
                .agg({"fx": "sum", "fy": "sum", "fz": "sum"})
                .min(axis=1)
                * FORCE_CONSTANT
            )

            # Extract the atoms with minimum force less than the maximum force specified
            # by the user
            min_force_atoms = (
                min_force[min_force < max_force]
                .reset_index()[["mol_type", "mol_id"]]
                .merge(neighbor_df, on=["mol_type", "mol_id"])
            )

            # Find the corresponding mol_id and mol_type for atom i
            atom_i_data = df[df["id"] == i].iloc[0]
            mol_id_corresponding_to_i = atom_i_data["mol_id"]
            mol_type_corresponding_to_i = atom_i_data["mol_type"]

            # Extract data for atom i and its corresponding mol
            data_head_i = atom_i_data[["id", "x", "y", "z"]].values

            # Extract data for the corresponding mol (excluding atom i)
            data_head_mol = min_force_atoms[
                (min_force_atoms["mol_id"] == mol_id_corresponding_to_i)
                & (min_force_atoms["mol_type"] == mol_type_corresponding_to_i)
                & (min_force_atoms["id"] != i)
            ][["id", "x", "y", "z"]].values

            # Extract data for all other atoms (excluding atom i and its mol)
            data_all = min_force_atoms[
                (min_force_atoms["mol_id"] != mol_id_corresponding_to_i)
                | (min_force_atoms["mol_type"] != mol_type_corresponding_to_i)
            ][["id", "x", "y", "z"]].values

            # Combine data for atom i, its mol, and all other atoms (all this to place
            # atom i and the molecule it corresponds to at the top of the data)
            data_all = np.vstack((data_head_i, data_head_mol, data_all))

            # Remove the boundary effects from the filtered atoms
            data_all = _remove_boundary_effects(data_head_i, data_all, lx, ly, lz, 1)

            # Write the clusters to *.xyz files
            fin_df = (
                pd.DataFrame(data_all, columns=["id", "x", "y", "z"])
                .merge(min_force_atoms[["element", "id"]], on="id")
                .drop("id", axis=1)
                .set_index("element")
                .reset_index()
            )
            frame_number = "{}{}".format(
                "0" * (len(str(len(dumps))) - len(str(index))), index
            )
            filename = "Cluster_{}_{}{}.xyz".format(
                frame_number,
                "0" * (len(str(len(atoms))) - len(str(counter - 1))),
                counter - 1,
            )
            f = open((os.path.join(working_dir, filename)), "w")
            f.write("{}\n\n".format(len(fin_df)))
            fin_df.to_csv(
                f, header=False, index=False, sep="\t", float_format="%15.10f"
            )
            f.close()
            cluster_count += 1
    return cluster_count


def get_unique_configurations(
    cluster_pattern,
    r_cut,
    molecules,
    mol_num,
    type_coord_atoms=None,
    working_dir=None,
    find_top=True,
    perc=None,
    cum_perc=90,
    mol_names=None,
    zip=True,
):
    """
    Identifies the configuration corresponding to each cluster file generated by the
    get_clusters function. Optionally, identifies the top configurations and saves them
    as conf_*.xyz files in the working directory. The clusters are assumed to be
    generated using the get_clusters function (this is because the atom of interest
    is assumed to be the first atom in the cluster file and the molecule to which this
    atom belongs is placed in the beginning of the cluster file; this format is followed
    in the get_clusters function). When identifying unique configurations,
    the clusters are grouped based on the number of molecules of each type in the
    cluster and the coordinating atoms in each cluster. The top configurations are
    determined based on one of two criteria: (1) ones that occur more than a certain
    percentage of the time or (2) ones that occur cumulatively more than a certain
    percentage of the time. The user can specify either criterion by providing the
    perc or cum_perc arguments, respectively. If both are provided, the cum_perc
    argument is used. Two to three csv files are generated: clusters.csv,
    configurations.csv, and top_conf.csv. top.conf.csv is only saved if find_top is set
    to True. The clusters.csv file contains the number of molecules of each type
    and the coordinating atoms in each cluster. The configurations.csv file contains the
    number and % of clusters for each unique configuration. The top_conf.csv file
    contains the top configurations and a corresponding sample cluster file for each
    unique configuration. A zip file containing all the original cluster files is also
    generated if zip is set to True.

    Args:
        cluster_pattern (str): the pattern of the cluster files, e.g. Cluster_*.xyz in
            the working_dir
        r_cut (float): the cutoff distance for two atoms to be considered coordinating
        molecules (list of pymatgen.core.structure.Molecule): the molecules in the
            system used to prepare the initial configuration for the LAMMPS simulations;
            should be in the same order they appear in the LAMMPS dump files
        mol_num (int): the order number of the molecule to which the atom of interest
            belongs as it appears in the molecules list; for example, if molecules is
            [dme, tfsi, mg], and the atom of interest belongs to tfsi, then mol_num is
            1 (0-based index)
        type_coord_atoms (list of str, optional): the type of the coordinating atoms to
            be considered in the clusters; defaults to None if all coordinating atoms
            are to be considered (i.e., no filtering)
        working_dir (str, optional): path to the working directory where the cluster
            files are located and the final csv files will be saved; defaults to the
            current working directory
        find_top (bool, optional): if True, the top configurations are determined based
            on either the percentage of the total number of clusters or the cumulative
            percentage of the total number of clusters provided by the user;
            defaults to True
        perc (float, optional): will return the unique configurations that have a
            frequency of occurrence greater than or equal to this value; defaults to None
        cum_perc (float, optional): will return the unique configurations that have a
            cumulative frequency of occurrence about this value; defaults to 90%,
            meaning that clusters that make up 90% of the total number of clusters will
            be returned if find_top is set to True
        mol_names (list of str, optional): the names of the molecules in the system;
            used when creating the final dataframes to mark the number of coordinating
            molecules of each type (e.g. num_x, num_y, etc.) and the coordinating atoms
            from each type (e.g. atoms_x, atoms_y, etc.); should match the order of the
            molecules attribute; defaults to None meaning that the naming will be done
            using the index of the molecules (e.g. num_1, num_2, etc. and atoms_1,
            atoms_2, etc.)
        zip (bool, optional): if True, a zip file (Clusters.zip) containing all the
            original cluster files will be generated to clean up the working_dir;
            defaults to True

    Returns:
        tuple: a tuple containing the following two dataframes: clusters and
            configurations

    Note:
        Be advised that the algorithm for identifying unique configurations may not
        accurately work in the following edge case. It is recommended to check your
        results when your system satisfies both of these conditions:
        - One main molecule type is a subset of another, with both types beginning with
          the same sequence of shared atom types, and
        - The smaller molecule type (the subset) precedes the larger molecule within
          the same cluster (this happens when the molecules are not in a decreasing
          order of their size in the original LAMMPS data/trajectory files)
    """
    working_dir = working_dir or os.getcwd()

    # Get a list of the cluster files
    cluster_files = glob.glob(f"{working_dir}/{cluster_pattern}")

    # Get a list of the atoms in each molecule
    main_atoms = []
    for mol in molecules:
        main_atoms.append([str(i) for i in mol.species])

    # Iterate over the cluster files and process each one
    full_coord_mols = {"cluster": [], "num_mols": [], "coordinating_atoms": []}
    for file_num, file in enumerate(
        tqdm(cluster_files, desc="Processing cluster files")
    ):
        mol = Molecule.from_file(file)
        full_coord_mols["cluster"].append(ntpath.basename(file))

        # Get the coordinating atoms to the fist atom in each cluster; it is assumed
        # that the first atom in each cluster is the atom of interest (this is the case
        # when using the get_clusters function)
        coord_atoms = mol.get_neighbors(mol[0], r_cut)

        # Filter for coordinating atoms of the type specified by the user
        if coord_atoms and type_coord_atoms:
            coord_atoms = [
                i for i in coord_atoms if i.species_string in type_coord_atoms
            ]

        # Get a list of the atoms in each cluster excluding the ones in the molecule
        # to which the atom of interest belongs
        cluster_atoms = [str(i) for i in mol.species][len(main_atoms[mol_num]) :]

        idx = 0
        coord_mols = {ind: {"mol": [], "sites": []} for ind in range(len(molecules))}
        while idx < len(cluster_atoms):
            for ind, atoms in enumerate(main_atoms):
                if cluster_atoms[idx: idx + len(atoms)] == atoms:
                    v_ = idx + len(main_atoms[mol_num])
                    sub_mol = mol[v_: v_ + len(atoms)]
                    idx += len(atoms)
                    coord_mols[ind]["mol"].append(sub_mol)
                    coords = []
                    for coord_atom in coord_atoms:
                        if coord_atom in sub_mol:
                            coords.append(coord_atom.species_string)
                    coord_mols[ind]["sites"].append(coords)
                    break
        for ind in coord_mols.keys():
            coord_mols[ind]["num_mol"] = len(coord_mols[ind]["mol"])
            del coord_mols[ind]["mol"]

        # Add the number of molecules and the coordinating atoms to the full dict
        full_coord_mols["num_mols"].append(
            list(coord_mols[k]["num_mol"] for k in coord_mols)
        )
        full_coord_mols["coordinating_atoms"].append(
            list(coord_mols[k]["sites"] for k in coord_mols)
        )

    # Cleanups and formatting for the output dataframes
    full_str_coord = []
    for i in full_coord_mols["coordinating_atoms"]:
        str_coord = []
        for j in i:
            str_full = []
            for k in j:
                c = dict(Counter(x[0] for x in k if x))
                str_full.append("".join(f"{c[k]}{k}" for k in sorted(c)))
            str_coord.append(":".join([i for i in sorted(str_full)]))
        full_str_coord.append(str_coord)
    full_coord_mols["coordinating_atoms"] = full_str_coord
    df = pd.DataFrame.from_dict(full_coord_mols, "columns")
    if mol_names:
        num_col_names = [f"num_{i}" for i in mol_names]
        atoms_col_names = [f"atoms_{i}" for i in mol_names]
    else:
        num_col_names = [f"num_{i+1}" for i in range(len(molecules))]
        atoms_col_names = [f"atoms_{i + 1}" for i in range(len(molecules))]
    split_df = pd.DataFrame(
        df["num_mols"].tolist(),
        columns=num_col_names,
    )
    df = pd.concat([df, split_df], axis=1)
    df = df.drop("num_mols", axis=1)
    split_df = pd.DataFrame(
        df["coordinating_atoms"].tolist(),
        columns=atoms_col_names,
    )
    df = pd.concat([df, split_df], axis=1)
    df = df.drop("coordinating_atoms", axis=1)
    df1 = (
        df.groupby([i for i in df.columns if i != "cluster"])
        .size()
        .rename("count")
        .reset_index()
    )
    df1.sort_values("count", ascending=False, inplace=True)
    df1["%"] = df1["count"] * 100 / sum(df1["count"])
    if find_top:
        if cum_perc and perc:
            warnings.warn(
                "Two percentage types are provided for determining the top "
                "configurations; using cum_perc"
            )
        if cum_perc:
            top_config = df1[df1["%"].cumsum() <= cum_perc]
        elif perc:
            top_config = df1[df1["%"] >= perc]
        else:
            raise ValueError(
                "No percentage type is provided for determining the top "
                "configurations"
            )
        merge_cols = [i for i in list(df.columns) if i.startswith("atoms_")]
        df = df.sort_values("cluster").reset_index(drop=True)
        top_config = top_config.merge(
            df[["cluster"] + merge_cols], on=merge_cols
        ).drop_duplicates(merge_cols)
        for ind, cluster in enumerate(top_config["cluster"]):
            shutil.copy(f"{working_dir}/{cluster}", f"{working_dir}/conf_{ind+1}.xyz")
        top_config.to_csv(f"{working_dir}/top_conf.csv", index=False)
    df.to_csv(f"{working_dir}/clusters.csv", index=False)
    df1.to_csv(f"{working_dir}/configurations.csv", index=False)
    if zip:
        clusters_dir = f"{working_dir}/Clusters"
        os.mkdir(f"{working_dir}/Clusters")
        for file in cluster_files:
            shutil.move(file, f"{clusters_dir}/{ntpath.basename(file)}")
        shutil.make_archive(f"{working_dir}/Clusters", "zip", clusters_dir)
        shutil.rmtree(clusters_dir)
    return df, df1
