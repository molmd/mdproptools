#!/usr/bin/env python

"""
Extracts clusters within a cutoff distance from an atom from LAMMPS trajectory files.
"""

import os
import glob
import ntpath
import shutil
import warnings

from collections import Counter

import numpy as np
import pandas as pd

from pymatgen.core.structure import Molecule
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from pymatgen.core.periodic_table import _pt_data
from pymatgen.analysis.molecule_matcher import MoleculeMatcher

from mdproptools.structural.rdf_cn import _calc_rsq, _calc_atom_type

__author__ = "Rasha Atwi, Maxim Makeev"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "Apr 2020"
__version__ = "0.0.1"

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
    full_trajectory=False,
    frame=None,
    num_mols=None,
    num_atoms_per_mol=None,
    elements=None,
    alter_atom_ids=False,
    max_force=0.75,
    working_dir=None,
):
    if elements:
        elements = {i + 1: j for i, j in enumerate(elements)}
    if not working_dir:
        working_dir = os.getcwd()
    dumps = list(parse_lammps_dumps(filename))
    if full_trajectory:
        dumps = dumps
    else:
        dumps = [dumps[frame]]

    cluster_count = 0
    for index, dump in enumerate(dumps):
        print("Processing frame number: {}".format(index))
        lx = dump.box.bounds[0][1] - dump.box.bounds[0][0]
        ly = dump.box.bounds[1][1] - dump.box.bounds[1][0]
        lz = dump.box.bounds[2][1] - dump.box.bounds[2][0]
        df = dump.data.sort_values(by=["id"])
        mol_types = []
        mol_ids = []
        for mol_type, number_of_mols in enumerate(num_mols):
            for mol_id in range(number_of_mols):
                for atom_id in range(num_atoms_per_mol[mol_type]):
                    mol_types.append(mol_type + 1)
                    mol_ids.append(mol_id + 1)
        df["mol_type"] = mol_types
        df["mol_id"] = mol_ids
        if elements:
            df["element"] = df["type"].map(elements)
        if alter_atom_ids:
            data = _calc_atom_type(df.values, num_mols, num_atoms_per_mol)
            df["type"] = data[:, 0]
        counter = 0
        atoms = df[df["type"] == atom_type]["id"]
        for i in atoms:
            counter += 1
            print("Processing atom number: {}".format(counter - 1))
            data_head = df[df["id"] == i][["mol_type", "mol_id", "x", "y", "z"]].values[
                0
            ]
            data_i, rsq = _calc_rsq(
                data_head,
                df.loc[:, ["mol_type", "mol_id", "x", "y", "z"]].values,
                lx,
                ly,
                lz,
                2,
            )
            cond = rsq < r_cut ** 2
            data_i = data_i[cond, :]
            ids = pd.DataFrame(
                np.unique(data_i[:, [0, 1]], axis=0), columns=["mol_type", "mol_id"]
            )
            neighbor_df = ids.merge(df, on=["mol_type", "mol_id"])
            min_force = (
                neighbor_df.groupby(["mol_type", "mol_id"])
                .agg({"fx": "sum", "fy": "sum", "fz": "sum"})
                .min(axis=1)
                * FORCE_CONSTANT
            )
            min_force_atoms = (
                min_force[min_force < max_force]
                .reset_index()[["mol_type", "mol_id"]]
                .merge(neighbor_df, on=["mol_type", "mol_id"])
            )
            # if elements:
            # min_force_atoms['element'] = min_force_atoms['type'].map(elements)
            data_head = df[df["id"] == i][["id", "x", "y", "z"]].values[0]
            data_i = min_force_atoms.loc[
                min_force_atoms["id"] != i, ["id", "x", "y", "z"]
            ].values
            data_i = np.insert(data_i, 0, data_head, axis=0)
            data_i = _remove_boundary_effects(data_head, data_i, lx, ly, lz, 1)
            fin_df = (
                pd.DataFrame(data_i, columns=["id", "x", "y", "z"])
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
            f = open((os.path.join(working_dir, filename)), "a")
            f.write("{}\n\n".format(len(fin_df)))
            fin_df.to_csv(
                f, header=False, index=False, sep="\t", float_format="%15.10f"
            )
            f.close()
            cluster_count += 1
        print(
            "{} clusters written to *.xyz files".format(
                len(df[df["type"] == atom_type]["id"])
            )
        )
    return cluster_count


def get_unique_configurations(
    cluster_pattern,
    r_cut,
    molecules,
    type_coord_atoms=None,
    working_dir=None,
    find_top=True,
    perc=None,
    cum_perc=90,
    mol_names=None,
    zip=True,
):
    working_dir = working_dir or os.getcwd()
    cluster_files = glob.glob(f"{working_dir}/{cluster_pattern}")
    main_atoms = []
    for mol in molecules:
        main_atoms.append([str(i) for i in mol.species])
    full_coord_mols = {"cluster": [], "num_mols": [], "coordinating_atoms": []}
    for file in cluster_files:
        mol = Molecule.from_file(file)
        full_coord_mols["cluster"].append(ntpath.basename(file))
        coord_atoms = mol.get_neighbors(mol[0], r_cut)
        if type_coord_atoms:
            coord_atoms = [
                i for i in coord_atoms if i.species_string in type_coord_atoms
            ]
        cluster_atoms = [str(i) for i in mol.species]
        coord_mols = {}
        for ind, atoms in enumerate(main_atoms):
            coord_mols[ind] = {"mol": [], "sites": []}
            for idx in range(len(cluster_atoms)):
                if cluster_atoms[idx : idx + len(atoms)] == atoms:
                    sub_mol = mol[idx : idx + len(atoms)]
                    coord_mols[ind]["mol"].append(sub_mol)
            for idx, sub_mol in enumerate(coord_mols[ind]["mol"]):
                coords = []
                for coord_atom in coord_atoms:
                    if coord_atom in sub_mol:
                        coords.append(coord_atom.species_string)
                coord_mols[ind]["sites"].append(coords)
            coord_mols[ind]["num_mol"] = len(coord_mols[ind]["mol"])
            del coord_mols[ind]["mol"]
        full_coord_mols["num_mols"].append(
            list(coord_mols[k]["num_mol"] for k in coord_mols)
        )
        full_coord_mols["coordinating_atoms"].append(
            list(coord_mols[k]["sites"] for k in coord_mols)
        )

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
    split_df = pd.DataFrame(df["num_mols"].tolist(), columns=num_col_names,)
    df = pd.concat([df, split_df], axis=1)
    df = df.drop("num_mols", axis=1)
    split_df = pd.DataFrame(df["coordinating_atoms"].tolist(), columns=atoms_col_names,)
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


def group_clusters(cluster_pattern, tolerance=0.1, working_dir=None):
    if not working_dir:
        working_dir = os.getcwd()
    mm = MoleculeMatcher(tolerance=tolerance)
    filename_list = glob.glob((os.path.join(working_dir, cluster_pattern)))
    mol_list = [Molecule.from_file(os.path.join(working_dir, f)) for f in filename_list]
    mol_groups = mm.group_molecules(mol_list)
    filename_groups = [
        [filename_list[mol_list.index(m)] for m in g] for g in mol_groups
    ]
    for p, i in enumerate(filename_groups):
        if p + 1 < 10:
            conf_num = "0{}".format(p + 1)
        else:
            conf_num = p + 1
        folder_name = "Configuration_{}".format(conf_num)
        if not os.path.exists(folder_name):
            os.mkdir((os.path.join(working_dir, folder_name)))
            for f in i:
                shutil.move(f, (os.path.join(working_dir, folder_name)))
        else:
            for f in i:
                shutil.move(f, (os.path.join(working_dir, folder_name)))
