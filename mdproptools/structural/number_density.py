#!/usr/bin/env python

"""
Calculates the number density from LAMMPS trajectory files.
"""

import os

from time import time
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from mdproptools.structural.rdf_cn import (
    _save_rdf,
    _initialize,
    _calc_atom_type,
    _define_dataframes,
)

__author__ = "Rasha Atwi, Maxim Makeev"
__maintainer__ = "Rasha Atwi"
__email__ = "rasha.atwi@stonybrook.edu"
__status__ = "Development"
__date__ = "May 2020"
__version__ = "0.0.1"


def calc_number_density(
    dump_pattern,
    surface_atom,
    atom_types,
    bin_size,
    dist_from_interface,
    axis_norm_interface,
    num_mols=None,
    num_atoms_per_mol=None,
    working_dir=None,
    results_file="number_density.csv",
    save_mode=True,
):
    # assumes the surface is composed of one atom type
    # TODO: allow center of mass calculations
    if not working_dir:
        working_dir = os.getcwd()
    partial_relations = np.array(
        (
            np.full(shape=len(atom_types), fill_value=surface_atom, dtype=np.int),
            atom_types,
        )
    )
    dumps, num_bins, radii, num_files, num_relations = _initialize(
        dist_from_interface,
        bin_size,
        os.path.join(working_dir, dump_pattern),
        partial_relations,
    )
    rho_part_sum = np.zeros((num_relations, num_bins))
    for dump in dumps:
        start_traj_loop, ref_df, _ = _define_dataframes(dump)

        # If number of molecules and number of atoms per molecule are provided,
        # calculate unique atom ids, else use the default ones
        if num_mols and num_atoms_per_mol:
            data = _calc_atom_type(ref_df.values, num_mols, num_atoms_per_mol)
            ref_df = pd.DataFrame(data, columns=["id", "type", "x", "y", "z"]).drop(
                "type", axis=1
            )
            atom_types_col = "id"
        else:
            ref_df = ref_df.drop("id", axis=1)
            atom_types_col = "type"
        # Calculate the distance range occupied by the surface atom in the
        # provided direction
        min_dist = ref_df[ref_df[atom_types_col] == surface_atom][
            axis_norm_interface
        ].min()
        max_dist = ref_df[ref_df[atom_types_col] == surface_atom][
            axis_norm_interface
        ].max()
        dist_range = max_dist - min_dist
        rho_part = np.zeros((num_relations, num_bins))
        ref_df[axis_norm_interface] -= min_dist
        st = time()
        # Calculate density profiles of the input atom types
        for i, j in enumerate(atom_types):
            b = ref_df[
                (ref_df[atom_types_col] == j)
                & (ref_df[axis_norm_interface] < dist_from_interface)
            ][axis_norm_interface].values
            b -= dist_range
            current_bin = (b / bin_size).astype(int)
            for k in current_bin:
                rho_part[i][k] += 1
        print("time:", time() - st)
        print("Finished computing density profile for timestep", dump.timestep)

        # Normalize the computed density profiles by dividing by bin volume
        box_lengths = dump.box.to_lattice().lengths
        box_lengths_dict = {
            "x": box_lengths[0],
            "y": box_lengths[1],
            "z": box_lengths[2],
        }
        rho_part = rho_part / (
            np.product(
                [
                    box_lengths_dict[k]
                    for k in box_lengths_dict
                    if k != axis_norm_interface
                ]
            )
            * bin_size
        )
        rho_part_sum += rho_part

        end_traj_loop = timer()
        print("Trajectory loop took:", end_traj_loop - start_traj_loop, "s")

    rho_part_sum = rho_part_sum / num_files
    final_df = _save_rdf(
        radii,
        np.asarray(partial_relations).transpose(),
        os.path.join(working_dir, results_file),
        save_mode,
        rho_part_sum,
    )
    return final_df
