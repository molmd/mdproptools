#!/usr/bin/env python

"""
Calculates RDF and coordination numbers from LAMMPS trajectory files.
"""

from time import time
from timeit import default_timer as timer

import numba
import numpy as np
import pandas as pd

from numba import njit, prange

from pymatgen.io.lammps.outputs import parse_lammps_dumps

"""
This module calculates radial distribution function (rdf) and coordination 
number (cn) from LAMMPS trajectory files.
"""

__author__ = "Rasha Atwi, Matthew Bliss, Maxim Makeev"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "Dec 2019"
__version__ = "0.0.1"

CON_CONSTANT = 1.660538921


# TODO: For the same atom repeating in multiple molecules, calculate RDF between
#  any atom and that atom in one specific molecule
@numba.jit(cache=True, nopython=True)  # , parallel=True)
def _calc_rsq(data_head, mol_data, lx, ly, lz, num_of_ids=1):
    # TODO: merge the new changes in this function with the ones in the most recent version
    """
    Calculates the squared distance between an atom and a set of atoms or
    molecules.
    """
    data_i = np.zeros((mol_data.shape[0], mol_data.shape[1] + 2))
    data_i[:, : num_of_ids + 3] = mol_data.copy()
    data_i[:, num_of_ids : num_of_ids + 3] = (
        data_head[num_of_ids:] - mol_data[:, num_of_ids:]
    )
    dx = data_i[:, num_of_ids]
    dy = data_i[:, num_of_ids + 1]
    dz = data_i[:, num_of_ids + 2]
    cond = (dx > lx / 2) | (dx < -lx / 2)
    dx[cond] = dx[cond] - np.sign(dx[cond]) * lx
    cond = (dy > ly / 2) | (dy < -ly / 2)
    dy[cond] = dy[cond] - np.sign(dy[cond]) * ly
    cond = (dz > lz / 2) | (dz < -lz / 2)
    dz[cond] = dz[cond] - np.sign(dz[cond]) * lz
    rsq = dx ** 2 + dy ** 2 + dz ** 2
    data_i[:, num_of_ids + 3] = rsq
    return data_i, rsq


@numba.jit(cache=True, nopython=True)  # , parallel=True)
def _remove_outliers(data_i, rsq, r_cut, ddr):
    """
    Removes pairs with distances bigger than the cutoff radius.
    """
    cond = rsq < r_cut ** 2
    data_i = data_i[cond, :]
    data_i[:, 5] = np.sqrt(data_i[:, 4]) / ddr
    return data_i


@numba.jit(cache=True, nopython=True)  # , parallel=True)
def _rdf_loop(
    data, relation_matrix, num_relations, lengths, r_cut, ddr, rdf_full, rdf_part
):
    """
    Calculates full and partial rdf between two atoms from a single LAMMPS
    trajectory.
    """
    lx, ly, lz = lengths
    for i in prange(0, data.shape[0] - 1):
        data_head = data[i, :]
        data_i, rsq = _calc_rsq(data_head, data[i + 1 :, :], lx, ly, lz)
        data_i = _remove_outliers(data_i, rsq, r_cut, ddr)
        for j in data_i[:, 5].astype(np.int64):
            rdf_full[j] += 2
        for kl in prange(0, num_relations):
            nta1, nta2 = relation_matrix[kl]
            if int(data_head[0]) == nta1:
                v_j = data_i[data_i[:, 0].astype(np.int64) == nta2]
                for j in v_j[:, 5].astype(np.int64):
                    rdf_part[kl][j] += 1
            if int(data_head[0]) == nta2:
                v_j = data_i[data_i[:, 0].astype(np.int64) == nta1]
                for j in v_j[:, 5].astype(np.int64):
                    rdf_part[kl][j] += 1
    return rdf_full, rdf_part


@numba.jit(cache=True, nopython=True)  # , parallel=True)
def _cn_loop(data, relation_matrix, num_relations, lengths, r_cut, ddr, cn):
    """
    Calculates coordination number between two atoms from a single LAMMPS
    trajectory.
    """
    lx, ly, lz = lengths
    for i in prange(0, data.shape[0] - 1):
        data_head = data[i, :]
        data_i, rsq = _calc_rsq(data_head, data[i + 1 :, :], lx, ly, lz)
        for kl in prange(0, num_relations):
            nta1, nta2 = relation_matrix[kl]
            data_i_cond = _remove_outliers(data_i, rsq, r_cut[kl], ddr)
            if int(data_head[0]) == nta1:
                v_j = data_i_cond[data_i_cond[:, 0].astype(np.int64) == nta2]
                cn[kl] += len(v_j[:, 5])
            if int(data_head[0]) == nta2:
                v_j = data_i_cond[data_i_cond[:, 0].astype(np.int64) == nta1]
                cn[kl] += len(v_j[:, 5])
    return cn


@numba.jit(cache=True, nopython=True)  # , parallel=True)
def _rdf_mol_loop(
    atom_data, mol_data, relation_matrix, num_relations, lengths, r_cut, ddr, rdf_part
):
    """
    Calculates partial rdf between an atom and the center of mass of a molecule
    from a single LAMMPS trajectory.
    """
    lx, ly, lz = lengths
    for i in prange(0, atom_data.shape[0]):
        data_head = atom_data[i, :]
        data_i, rsq = _calc_rsq(data_head, mol_data, lx, ly, lz)
        data_i = _remove_outliers(data_i, rsq, r_cut, ddr)
        for kl in prange(0, num_relations):
            nta1, nta2 = relation_matrix[kl]
            if int(data_head[0]) == nta1:
                v_j = data_i[data_i[:, 0].astype(np.int64) == nta2]
                for j in v_j[:, -1].astype(np.int64):
                    rdf_part[kl][j] += 1
    return rdf_part


@numba.jit(cache=True, nopython=True)  # , parallel=True)
def _cn_mol_loop(
    atom_data, mol_data, relation_matrix, num_relations, lengths, r_cut, ddr, cn
):
    """
    Calculates coordination number between an atom and the center of mass of a
    molecule from a single LAMMPS trajectory.
    """
    lx, ly, lz = lengths
    for i in prange(0, atom_data.shape[0]):
        data_head = atom_data[i, :]
        data_i, rsq = _calc_rsq(data_head, mol_data, lx, ly, lz)
        for kl in prange(0, num_relations):
            nta1, nta2 = relation_matrix[kl]
            data_i_cond = _remove_outliers(data_i, rsq, r_cut[kl], ddr)
            if int(data_head[0]) == nta1:
                v_j = data_i_cond[data_i_cond[:, 0].astype(np.int64) == nta2]
                cn[kl] += len(v_j[:, -1])
    return cn


def _initialize(r_cut, bin_size, filename, partial_relations):
    """
    Defines inputs required for calculating rdf and coordination number.
    """
    if isinstance(r_cut, list):
        num_bins = [int(i / bin_size) for i in r_cut]
        radii = [(np.arange(i) + 0.5) * bin_size for i in num_bins]
    else:
        num_bins = int(r_cut / bin_size)
        radii = (np.arange(num_bins) + 0.5) * bin_size

    dumps = list(parse_lammps_dumps(filename))
    num_files = len(dumps)
    num_relations = len(partial_relations[0])

    return dumps, num_bins, radii, num_files, num_relations


def _define_dataframes(dump):
    """
    Defines the atom and object (atom or molecule) dataframes containing id,
    type, and 3d coordinates required for rdf and coordination number
    calculations.
    """
    start_traj_loop = timer()
    print("The timestep of the current file is: " + str(dump.timestep))
    ref_df = dump.data[["id", "type", "x", "y", "z"]]
    ref_df = ref_df.sort_values("id")
    df = ref_df.copy()
    return start_traj_loop, ref_df, df


def _calc_atom_type(data, num_mols, num_atoms):
    """
    Defines new atom ids based on the number of atoms in each molecule type.
    """
    total_num_atoms = np.multiply(num_mols, num_atoms)
    transformer = np.ones((len(total_num_atoms), len(total_num_atoms)), int)
    transformer = np.tril(transformer, 0)
    atom_type_cutoff = np.matmul(transformer, total_num_atoms)

    for n in range(np.shape(data)[0]):
        for i, cutoff in enumerate(atom_type_cutoff):
            if data[n][0] <= cutoff:
                data[n][0] = (data[n][0] - cutoff) % num_atoms[i]
                if data[n][0] == 0:
                    data[n][0] = num_atoms[i]
                if i > 0:
                    data[n][0] += np.sum(num_atoms[:i])
                break
    return data


def _define_mol_cols(df, num_mols, num_atoms_per_mol, mass):
    """
    Calculates the center of mass of each molecule.
    """
    mol_types = []
    mol_ids = []
    for mol_type, number_of_mols in enumerate(num_mols):
        for mol_id in range(number_of_mols):
            for atom_id in range(num_atoms_per_mol[mol_type]):
                mol_types.append(mol_type + 1)
                mol_ids.append(mol_id + 1)
    df["mol_type"] = mol_types
    df["mol_id"] = mol_ids
    df["mass"] = df.apply(lambda x: mass[int(x.type - 1)], axis=1)
    df = df.drop(["type", "id"], axis=1).set_index(["mol_type", "mol_id"])
    mol_df = df.groupby(["mol_type", "mol_id"]).apply(
        lambda x: pd.Series(
            x["mass"].values @ x[["x", "y", "z"]].values / x.mass.sum(),
            index=["x", "y", "z"],
        )
    )
    return (
        mol_df.reset_index().drop("mol_id", axis=1).rename(columns={"mol_type": "type"})
    )


def _calc_props(
    dump,
    ref_df,
    df,
    num_types,
    mass,
    num_relations,
    partial_relations,
    atom_types_col,
    num_atoms_per_mol=None,
):
    """
    Calculates densities (total mass, total number, and partial number) and box
    lengths required for rdf and coordination number calculations.
    """
    n_objects = df.shape[0]
    box_lengths = dump.box.to_lattice().lengths
    volume = np.prod(box_lengths)
    atom_types = ref_df[atom_types_col].astype(np.int64).value_counts().to_dict()
    object_types = df[atom_types_col].astype(np.int64).value_counts().to_dict()
    set_id = set(atom_types.keys())
    if atom_types_col == "type":
        if num_types != len(set_id):
            raise ValueError(
                f"""Consistency check failed: Number of specified 
                            atomic types is different from the calculated value 
                            specified= {num_types}, calculated= {len(set_id)}"""
            )
    elif atom_types_col == "id":
        if np.sum(num_atoms_per_mol) != len(set_id):
            raise ValueError(
                f"""Consistency check failed: Number of specified 
                            atomic types is different from the calculated value 
                            specified= {num_atoms_per_mol}, 
                            calculated= {len(set_id)}"""
            )
    total_mass = np.sum(
        [float(mass[i]) * float(atom_types[i + 1]) for i in range(num_types)]
    )
    print(total_mass)
    total_density = float((total_mass / volume) * CON_CONSTANT)
    print(total_density)
    print("{0:s}{1:10.8f}".format("Average density=", float(total_density)))

    rho = n_objects / volume
    rho_pairs = np.zeros(num_relations)
    for index, object_type in enumerate(partial_relations[1]):
        rho_pairs[index] = object_types[object_type] / volume
        if rho_pairs[index] < 1.0e-22:
            raise ValueError("Error: Density is zero for mol type: " + str(object_type))
    return rho, rho_pairs, box_lengths, atom_types, object_types


def _normalize_rdf(
    bin_size,
    rho_pairs,
    atom_types,
    partial_relations,
    num_relations,
    num_bins,
    rdf_part,
    rdf_full=None,
    ref_df=None,
    rho=None,
):
    """
    Normalizes the full and partial rdf (atom-atom or atom-molecule).
    """
    shell_volume = (
        4
        / 3
        * np.pi
        * bin_size ** 3
        * (np.arange(1, num_bins + 1) ** 3 - np.arange(num_bins) ** 3)
    )
    if rdf_full is not None:
        num_atoms = ref_df.shape[0]
        rdf_full = rdf_full / (num_atoms * rho * shell_volume)

    ref_atoms = np.asarray(partial_relations[0]).reshape((num_relations, 1))
    ref_atoms_matrix = np.tile(ref_atoms, num_bins)
    num_atoms_matrix = np.vectorize(atom_types.get)(ref_atoms_matrix)
    rho_pairs_matrix = np.tile(rho_pairs.reshape((num_relations, 1)), num_bins)
    shell_volume_matrix = np.tile(shell_volume, (num_relations, 1))
    rdf_part = rdf_part / (num_atoms_matrix * rho_pairs_matrix * shell_volume_matrix)
    return rdf_full, rdf_part


def _normalize_cn(atom_types, partial_relations, cn):
    """
    Normalizes the coordination number (atom-atom or atom-molecule).
    """
    num_ref_atoms = [atom_types[i] for i in partial_relations[0]]
    cn = cn / num_ref_atoms
    return cn


def _save_rdf(
    radii, relation_matrix, path_or_buf, save_mode, rdf_part_sum, rdf_full_sum=None
):
    """
    Saves the computed full and partial rdf (atom-atom or atom-molecule) into a
    pd.DataFrame (and csv file if save_mode is True).
    """
    if rdf_full_sum is not None:
        result_tuple = (radii, rdf_full_sum, rdf_part_sum)
        final_label = ["r ($\AA$)", "g_full(r)"]
    else:
        result_tuple = (radii, rdf_part_sum)
        final_label = ["r ($\AA$)"]
    final_array = np.vstack(result_tuple).transpose()
    final_labels = final_label + [
        f"g_{str(pair[0])}-{str(pair[1])}" for pair in relation_matrix
    ]
    final_df = pd.DataFrame(final_array, columns=final_labels)
    if save_mode:
        final_df.to_csv(path_or_buf, index=False)
        print("Results are written to pd.DataFrame and csv file")
    else:
        print(final_df)
        print("Results are written to pd.DataFrame")
    return final_df


def _save_cn(relation_matrix, path_or_buff, cn_sum, save_mode):
    """
    Saves the coordination number (atom-atom or atom-molecule) into a
    pd.DataFrame (and csv file if save_mode is True).
    """
    final_array = np.vstack(cn_sum).transpose()
    final_labels = [f"cn_{str(pair[0])}-{str(pair[1])}" for pair in relation_matrix]
    final_df = pd.DataFrame(final_array, columns=final_labels)
    if save_mode:
        final_df.to_csv(path_or_buff, index=False)
        print("CN results are written to pd.DataFrame and csv file")
    else:
        print(final_df)
        print("CN results are written to pd.DataFrame")
    return final_df


def calc_atomic_rdf(
    r_cut,
    bin_size,
    num_types,
    mass,
    partial_relations,
    filename,
    num_mols=None,
    num_atoms_per_mol=None,
    path_or_buff="rdf.csv",
    save_mode=True,
):
    """
    Calculates full and partial rdf between two atoms based on LAMMPS dump
    files. Assumes 3d coordinates are dumped. Uses either default atom ids or
    recalculated ids based on the number of atoms in each molecule type if
    num_mols and num_atoms_per_mol are input. Works for both single and multiple
    LAMMPS dump files. Saves the output in a pd.DataFrame (and *.csv file if
    save_mode is True).

    Args:
        r_cut (float): maximum allowed distance (in LAMMPS units) after
            which a pair of atoms is discarded
        bin_size (float): width of the bins (in LAMMPS units)
        num_types (int): the number of unique atom types in a LAMMPS dump file
        mass (list of float): the mass of unique atom types in a LAMMPS dump
            file; should be in the same order as in the LAMMPS data file
        partial_relations (list of list of int): the reference atom types in the
            first list and the corresponding atom types in the second list; for
            example: [[8, 7], [1, 3]] calculates the partial rdf between 8 and
            1 and the partial rdf between 7 and 3
        filename (str or file handle): the name of the LAMMPS dump file; can be
            the entire name for a single file or a file pattern with the
            wildcard character ('*') for multiple dumps
        num_mols (list of int, optional): the number of molecules of each
            molecule type; should be consistent with PackmolRunner input;
            required if new atom ids need to be calculated to distinguish
            between two similar atoms in the same molecule or a different
            molecule; defaults to None if default LAMMPS atom ids are to be used
        num_atoms_per_mol (list of int, optional): the number of atoms in each
            molecule type; required if new atom ids need to be calculated to
            distinguish between two similar atoms in the same molecule or a
            different molecule; defaults to None if default LAMMPS atom ids are
            to be used
        path_or_buff (str or file handle): file path or object to which to save
            the full and partial rdf results; if nothing is specified, rdf
            results are saved into an rdf.csv file in the same working directory
        save_mode (bool): if True, rdf results are saved to csv file; otherwise,
            rdf results are saved to a pd.DataFrame; defaults to True

    Returns:
        pd.DataFrame containing radii, full rdf, and partial rdf

    Examples:
    >>> rdf_default_ids = calc_atomic_rdf(20, 0.05, 8, [16, 12.01, 1.008,\
                                        14.01, 32.06, 16, 19, 24.305],\
                                        [[8, 8, 8, 8], [1, 4, 6, 8]], \
                                        'Tests/Mg_2TFSI_G1.lammpstrj.*',\
                                        path_or_buff='Tests/rdf_default_ids.csv')

    >>> rdf_altered_ids = calc_atomic_rdf(20, 0.05, 8, [16, 12.01, 1.008,\
                                        14.01, 32.06, 16, 19, 24.305],\
                                        [[32, 32], [17, 32]],\
                                        'Tests/Mg_2TFSI_G1.lammpstrj.*',\
                                        num_mols=[591, 66, 33], \
                                        num_atoms_per_mol=[16, 15, 1],\
                                        path_or_buff='Tests/rdf_altered_ids.csv')
    """
    dumps, num_bins, radii, num_files, num_relations = _initialize(
        r_cut, bin_size, filename, partial_relations
    )
    rdf_full_sum = np.zeros(num_bins)
    rdf_part_sum = np.zeros((num_relations, num_bins))

    for dump in dumps:
        start_traj_loop, ref_df, _ = _define_dataframes(dump)
        # TODO: checking step
        if num_mols and num_atoms_per_mol:
            data = _calc_atom_type(ref_df.values, num_mols, num_atoms_per_mol)
            ref_df = pd.DataFrame(data, columns=["id", "type", "x", "y", "z"]).drop(
                "type", axis=1
            )
            atom_types_col = "id"
        else:
            ref_df = ref_df.drop("id", axis=1)
            atom_types_col = "type"

        rho, rho_pairs, box_lengths, atom_types, object_types = _calc_props(
            dump,
            ref_df,
            ref_df,
            num_types,
            mass,
            num_relations,
            partial_relations,
            atom_types_col,
            num_atoms_per_mol=num_atoms_per_mol,
        )

        relation_matrix = np.asarray(partial_relations).transpose()
        rdf_full = np.zeros(num_bins)
        rdf_part = np.zeros((num_relations, num_bins))
        st = time()
        ref_data = ref_df.values
        rdf_full, rdf_part = _rdf_loop(
            ref_data,
            relation_matrix,
            num_relations,
            box_lengths,
            r_cut,
            bin_size,
            rdf_full,
            rdf_part,
        )
        print("time:", time() - st)
        print("Finished computing RDF for timestep", dump.timestep)

        rdf_full, rdf_part = _normalize_rdf(
            bin_size,
            rho_pairs,
            atom_types,
            partial_relations,
            num_relations,
            num_bins,
            rdf_part,
            rdf_full,
            ref_df,
            rho,
        )
        rdf_full_sum += rdf_full
        rdf_part_sum += rdf_part

        end_traj_loop = timer()
        print("Trajectory loop took:", end_traj_loop - start_traj_loop, "s")

    rdf_full_sum = rdf_full_sum / num_files
    rdf_part_sum = rdf_part_sum / num_files
    final_df = _save_rdf(
        radii,
        relation_matrix,
        path_or_buff,
        save_mode,
        rdf_part_sum,
        rdf_full_sum=rdf_full_sum,
    )
    return final_df


def calc_atomic_cn(
    r_cut,
    bin_size,
    num_types,
    mass,
    partial_relations,
    filename,
    num_mols=None,
    num_atoms_per_mol=None,
    path_or_buff="cn.csv",
    save_mode=True,
):
    """
    Calculates coordination number between two atoms based on LAMMPS dump files.
    Assumes 3d coordinates are dumped. Uses either default atom ids or
    recalculated ids based on the number of atoms in each molecule type if
    num_mols and num_atoms_per_mol are input. Works for both single and multiple
    LAMMPS dump files. Saves the output in a pd.DataFrame (and *.csv file is
    save_mode is True).

    Args:
        r_cut (list of float): maximum allowed distance (in LAMMPS units)
            between two atoms after which a pair of atoms is discarded; each
            r_cut in the list corresponds to one pair of atoms
        bin_size (float): width of the bins (in LAMMPS units)
        num_types (int): the number of unique atom types in a LAMMPS dump file
        mass (list of float): the mass of unique atom types in a LAMMPS dump
            file; should be in the same order as in the LAMMPS data file
        partial_relations (list of list of int): the reference atom types in the
            first list and the corresponding atom types in the second list; for
            example: [[8, 7], [1, 3]] calculates the cn between 8 and 1 and the
            cn between 7 and 3
        filename (str or file handle): the name of the LAMMPS dump file; can be
            the entire name for a single file or a file pattern with the
            wildcard character ('*') for multiple dumps
        num_mols (list of int, optional): the number of molecules of each
            molecule type; should be consistent with PackmolRunner input;
            required if new atom ids need to be calculated to distinguish
            between two similar atoms in the same molecule or a different
            molecule; defaults to None if default LAMMPS atom ids are to be used
        num_atoms_per_mol (list of int, optional): the number of atoms in each
            molecule type; required if new atom ids need to be calculated to
            distinguish between two similar atoms in the same molecule or a
            different molecule; defaults to None if default LAMMPS atom ids
            are to be used
        path_or_buff (str or file handle): file path or object to which to save
            the cn results; if nothing is specified, cn results are saved into
            an cn.csv file in the same working directory
        save_mode (bool): if True, cn results are saved to csv file; otherwise,
            cn results are saved to a pd.DataFrame; defaults to True

    Returns:
        pd.DataFrame containing radii and cn

    Examples:
    >>> cn_default_ids = calc_atomic_cn([2.325, 4.375, 2.375, 13.0], 0.05, 8, \
                                    [16, 12.01, 1.008, 14.01, 32.06, 16, 19, \
                                    24.305], [[8, 8, 8, 8], [1, 4, 6, 8]], \
                                    'Tests/Mg_2TFSI_G1.lammpstrj.*',\
                                    path_or_buff='Tests/cn_default_ids.csv')

    >>> cn_altered_ids = calc_atomic_cn([4.375, 13.0], 0.05, 8, [16, 12.01, \
                                        1.008, 14.01, 32.06, 16, 19, 24.305],\
                                        [[32, 32], [17, 32]], \
                                        'Tests/Mg_2TFSI_G1.lammpstrj.*',\
                                        num_mols=[591, 66, 33], \
                                        num_atoms_per_mol=[16, 15, 1],\
                                        path_or_buff='Tests/cn_altered_ids.csv')
    """
    dumps, num_bins, radii, num_files, num_relations = _initialize(
        r_cut, bin_size, filename, partial_relations
    )
    cn_sum = np.zeros(num_relations)

    for dump in dumps:
        start_traj_loop, ref_df, _ = _define_dataframes(dump)
        # TODO: checking step
        if num_mols and num_atoms_per_mol:
            data = _calc_atom_type(ref_df.values, num_mols, num_atoms_per_mol)
            ref_df = pd.DataFrame(data, columns=["id", "type", "x", "y", "z"]).drop(
                "type", axis=1
            )
            atom_types_col = "id"
        else:
            ref_df = ref_df.drop("id", axis=1)
            atom_types_col = "type"

        rho, rho_pairs, box_lengths, atom_types, object_types = _calc_props(
            dump,
            ref_df,
            ref_df,
            num_types,
            mass,
            num_relations,
            partial_relations,
            atom_types_col,
            num_atoms_per_mol=num_atoms_per_mol,
        )

        relation_matrix = np.asarray(partial_relations).transpose()
        cn = np.zeros(num_relations)
        st = time()
        ref_data = ref_df.values
        cn = _cn_loop(
            ref_data, relation_matrix, num_relations, box_lengths, r_cut, bin_size, cn
        )

        print("time:", time() - st)
        print("Finished computing CN for timestep", dump.timestep)

        cn = _normalize_cn(atom_types, partial_relations, cn)
        cn_sum += cn

        end_traj_loop = timer()
        print("Trajectory loop took:", end_traj_loop - start_traj_loop, "s")

    cn_sum = cn_sum / num_files
    final_df = _save_cn(relation_matrix, path_or_buff, cn_sum, save_mode)
    return final_df


def calc_molecular_rdf(
    r_cut,
    bin_size,
    num_types,
    mass,
    partial_relations,
    filename,
    num_mols,
    num_atoms_per_mol,
    path_or_buff="rdf_mol.csv",
    save_mode=True,
):
    """
    Calculates partial rdf between an atom and the center of mass of a molecule
    based on LAMMPS dump files. Assumes 3d coordinates are dumped. Works for
    both single and multiple LAMMPS dump files. Saves the output in a
    pd.DataFrame (and *.csv file if save_mode is True).

    Args:
        r_cut (float): maximum allowed distance (in LAMMPS units) after
            which a pair of atom and molecule is discarded
        bin_size (float): width of the bins (in LAMMPS units)
        num_types (int): the number of unique atom types in a LAMMPS dump file
        mass (list of float): the mass of unique atom types in a LAMMPS dump
            file; should be in the same order as in the LAMMPS data file
        partial_relations (list of list of int): the reference atom types in the
            first list and the corresponding molecule types in the second list;
            for example: [[8, 7], [1, 3]] calculates the partial rdf between
            atom 8 and com of molecule 1 and the partial rdf between atom 7 and
            com of molecule 3; molecule ids follow the same order as
            PackmolRunner input
        filename (str or file handle): the name of the LAMMPS dump file; can be
            the entire name for a single file or a file pattern with the
            wildcard character ('*') for multiple dumps
        num_mols (list of int): the number of molecules of each molecule type;
            should be consistent with PackmolRunner input
        num_atoms_per_mol (list of int): the number of atoms in each molecule
            type
        path_or_buff (str or file handle): file path or object to which to save
            the partial rdf results; if nothing is specified, rdf results are
            saved into a rdf_mol.csv file in the same working directory
        save_mode (bool): if True, rdf results are saved to csv file; otherwise,
            rdf results are saved to a pd.DataFrame; defaults to True

    Returns:
        pd.DataFrame containing radii and partial rdf

    Examples:
    >>> rdf_default_ids = calc_molecular_rdf(20, 0.05, 8, [16, 12.01, 1.008,\
                                        14.01, 32.06, 16, 19, 24.305],\
                                        [[8, 8, 4], [1, 2, 3]], \
                                        'Tests/Mg_2TFSI_G1.lammpstrj.*',\
                                        num_mols=[591, 66, 33], \
                                        num_atoms_per_mol=[16, 15, 1],\
                                        path_or_buff='Tests/rdf_mol.csv')
    """
    dumps, num_bins, radii, num_files, num_relations = _initialize(
        r_cut, bin_size, filename, partial_relations
    )
    rdf_part_sum = np.zeros((num_relations, num_bins))
    for dump in dumps:
        start_traj_loop, ref_df, df = _define_dataframes(dump)
        # TODO: checking step
        df = _define_mol_cols(df, num_mols, num_atoms_per_mol, mass)
        ref_df = ref_df.drop("id", axis=1)
        rho, rho_pairs, box_lengths, atom_types, object_types = _calc_props(
            dump, ref_df, df, num_types, mass, num_relations, partial_relations, "type"
        )
        relation_matrix = np.asarray(partial_relations).transpose()
        rdf_part = np.zeros((num_relations, num_bins))
        st = time()
        atom_data = ref_df.values
        mol_data = df.values
        rdf_part = _rdf_mol_loop(
            atom_data,
            mol_data,
            relation_matrix,
            num_relations,
            box_lengths,
            r_cut,
            bin_size,
            rdf_part,
        )
        print("time:", time() - st)
        print("Finished computing RDF for timestep", dump.timestep)

        _, rdf_part = _normalize_rdf(
            bin_size,
            rho_pairs,
            atom_types,
            partial_relations,
            num_relations,
            num_bins,
            rdf_part,
        )
        rdf_part_sum += rdf_part

        end_traj_loop = timer()
        print("Trajectory loop took:", end_traj_loop - start_traj_loop, "s")

    rdf_part_sum = rdf_part_sum / num_files
    final_df = _save_rdf(radii, relation_matrix, path_or_buff, save_mode, rdf_part_sum)
    return final_df


def calc_molecular_cn(
    r_cut,
    bin_size,
    num_types,
    mass,
    partial_relations,
    filename,
    num_mols,
    num_atoms_per_mol,
    path_or_buff="cn_mol.csv",
    save_mode=True,
):
    """
    Calculates coordination number between an atom and the center of mass of a
    molecule based on LAMMPS dump files. Assumes 3d coordinates are dumped.
    Works for both single and multiple LAMMPS dump files. Saves the output in
    a pd.DataFrame (and *.csv file if save_mode is True).

    Args:
        r_cut (list of float): maximum allowed distance (in LAMMPS units) after
            which a pair of atom and molecule is discarded; each r_cut in the
            list corresponds to one pair of atom and molecule
        bin_size (float): width of the bins (in LAMMPS units)
        num_types (int): the number of unique atom types in a LAMMPS dump file
        mass (list of float): the mass of unique atom types in a LAMMPS
            dump file; should be in the same order as in the LAMMPS data file
        partial_relations (list of list of int): the reference atom types in the
            first list and the corresponding molecule types in the second list;
            for example: [[8, 7], [1, 3]] calculates the cn between atom 8 and
            com of molecule 1 and the cn between atom 7 and com of molecule 3;
            molecule ids follow the same order as PackmolRunner input
        filename (str or file handle): the name of the LAMMPS dump file; can be
            the entire name for a single file or a file pattern with the
            wildcard character ('*') for multiple dumps
        num_mols (list of int): the number of molecules of each molecule type;
            should be consistent with PackmolRunner input
        num_atoms_per_mol (list of int): the number of atoms in each molecule
            type
        path_or_buff (str or file handle): file path or object to which to save
            the cn results; if nothing is specified, cn results are saved into
            a cn_mol.csv file in the same working directory
        save_mode (bool): if True, cn results are saved to csv file; otherwise,
            cn results are saved to a pd.DataFrame; defaults to True

    Returns:
        pd.DataFrame containing radii and partial rdf

    Examples:
    >>> rdf_default_ids = calc_molecular_cn([2.325, 3.775, 4.375], 0.05, 8,\
                                            [16, 12.01, 1.008, 14.01, 32.06, \
                                            16, 19, 24.305],[[8, 8, 4], \
                                            [1, 2, 3]], \
                                            'Tests/Mg_2TFSI_G1.lammpstrj.*',\
                                            num_mols=[591, 66, 33], \
                                            num_atoms_per_mol=[16, 15, 1],\
                                            path_or_buff='Tests/cn_mol.csv')
    """
    dumps, num_bins, radii, num_files, num_relations = _initialize(
        r_cut, bin_size, filename, partial_relations
    )
    cn_sum = np.zeros(num_relations)

    for dump in dumps:
        start_traj_loop, ref_df, df = _define_dataframes(dump)
        # TODO: checking step
        df = _define_mol_cols(df, num_mols, num_atoms_per_mol, mass)
        ref_df = ref_df.drop("id", axis=1)
        rho, rho_pairs, box_lengths, atom_types, object_types = _calc_props(
            dump, ref_df, df, num_types, mass, num_relations, partial_relations, "type"
        )
        relation_matrix = np.asarray(partial_relations).transpose()
        cn = np.zeros(num_relations)
        st = time()
        atom_data = ref_df.values
        mol_data = df.values
        cn = _cn_mol_loop(
            atom_data,
            mol_data,
            relation_matrix,
            num_relations,
            box_lengths,
            r_cut,
            bin_size,
            cn,
        )
        print("time:", time() - st)
        print("Finished computing CN for timestep", dump.timestep)

        cn = _normalize_cn(atom_types, partial_relations, cn)
        cn_sum += cn

        end_traj_loop = timer()
        print("Trajectory loop took:", end_traj_loop - start_traj_loop, "s")

    cn_sum = cn_sum / num_files
    final_df = _save_cn(relation_matrix, path_or_buff, cn_sum, save_mode)
    return final_df

def calc_intermolecular_rdf(r_cut, bin_size, num_types, mass, partial_relations,
                            filename, num_mols, num_atoms_per_mol,
                            path_or_buff='rdf_mol.csv',
                            save_mode=True):
    # TODO: recheck this function, was written quickly for Li-S project
    # TODO: prevent calculating rdf between the mol and itself
    # from mdproptools.common.com_mols import calc_com
    dumps, num_bins, radii, num_files, num_relations = \
        _initialize(r_cut,
                    bin_size,
                    filename,
                    partial_relations)
    rdf_part_sum = np.zeros((num_relations, num_bins))
    for dump in dumps:
        start_traj_loop, ref_df, df = _define_dataframes(dump)
        df = _define_mol_cols(df, num_mols, num_atoms_per_mol, mass)
        # df = calc_com(df, num_mols, num_atoms_per_mol, mass, atom_attributes=["x", "y", "z"])
        # ref_df = ref_df.drop('id', axis=1)
        atom_types_col = 'type'
        rho, rho_pairs, box_lengths, atom_types, object_types = \
            _calc_props(dump, df, df, num_types, mass, num_relations,
                        partial_relations, atom_types_col,
                        num_atoms_per_mol=num_atoms_per_mol)

        relation_matrix = np.asarray(partial_relations).transpose()
        rdf_part = np.zeros((num_relations, num_bins))
        st = time()
        # atom_data = ref_df.values
        mol_data = df.values
        rdf_part = _rdf_mol_loop(mol_data, mol_data, relation_matrix,
                                 num_relations, box_lengths, r_cut, bin_size,
                                 rdf_part)
        print("time:", time() - st)
        print("Finished computing RDF for timestep", dump.timestep)

        _, rdf_part = _normalize_rdf(bin_size, rho_pairs, atom_types,
                                     partial_relations, num_relations, num_bins,
                                     rdf_part)
        rdf_part_sum += rdf_part

        end_traj_loop = timer()
        print('Trajectory loop took:', end_traj_loop - start_traj_loop, 's')

    rdf_part_sum = rdf_part_sum / num_files
    final_df = _save_rdf(radii, relation_matrix, path_or_buff, save_mode,
                         rdf_part_sum)
    return final_df

if __name__ == "__main__":
    print()
