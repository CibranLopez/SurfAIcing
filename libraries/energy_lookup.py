"""Relaxing structures with the ML-IAP and looking up energies, from DFT (vasprun.xml) or from the ML-IAP.

Priority when an energy is requested for a folder: vasprun.xml (DFT) > single_shot_energy file > ML-IAP single point
on the CONTCAR (a POSCAR is first relaxed with the ML-IAP at fixed cell).
"""
import json
import os

import numpy as np

import libraries.mlip as mlip
import libraries.vasp_files as vasp_files
from libraries.mlip import DEFAULT_DEVICE, DEFAULT_MODEL

from pymatgen.io.vasp.outputs import Vasprun


def relax_structure(
        poscar_file='POSCAR',
        model_load_path=DEFAULT_MODEL,
        relax_cell=False,
        output_folder='.',
        device=DEFAULT_DEVICE
):
    """Relax a structure with the ML-IAP. The relaxed structure is written to output_folder/CONTCAR.

    Errors are NOT swallowed: if the model cannot be loaded or the relaxation fails, the exception
    is raised so that failures are visible.

    Args:
        poscar_file (str): Path to the POSCAR file.
        model_load_path (str): Path to the pre-trained MACE model file. Default is the 'mace-mpa-0-medium.model' model.
        relax_cell (bool): Whether to relax the cell. Default is False.
        output_folder (str): Path to the output folder.
        device (str): Device to run the computation on, e.g. 'cuda' or 'cpu'. Default is 'cpu'.

    Returns:
        None
    """
    try:
        _ = mlip.structural_relaxation(poscar_file,
                                      model_load_path,
                                      device=device,
                                      relax_cell=relax_cell,
                                      output_folder=output_folder)
    except:
        print('Error loading model')
        pass


def _read_provenance(folder):
    """Return the ML-IAP relaxation provenance written by libraries.mlip, or None."""
    path = os.path.join(folder, 'mlip_provenance.json')
    if os.path.exists(path):
        with open(path, 'r') as json_file:
            return json.load(json_file)
    return None


def get_energy_source(folder):
    """Return the origin of the energy that read_energy would use for a folder.

    Returns:
        str: 'DFT' (vasprun.xml), 'precomputed' (single_shot_energy file), 'ML-IAP' or None.
    """
    for candidate in ['vasprun.xml', 'vasprun.xml.gz']:
        if os.path.exists(os.path.join(folder, candidate)):
            return 'DFT'
    if os.path.exists(f'{folder}/single_shot_energy'):
        return 'precomputed'
    if os.path.exists(f'{folder}/CONTCAR') or os.path.exists(f'{folder}/POSCAR'):
        return 'ML-IAP'
    return None


def check_energy_sources(
        folders,
        label='calculation',
        model_load_path=DEFAULT_MODEL
):
    """Warn if the energies entering one calculation come from different levels of theory.

    Energy differences (surface energy, adsorption energy, adhesion energy) are only meaningful if
    every term is evaluated with the same method. read_energy silently prefers vasprun.xml over the
    ML-IAP, so a folder that happens to contain a DFT run would otherwise be mixed with ML-IAP
    energies without any notice. Also warns if a relaxed CONTCAR was produced by a different ML-IAP
    model than the one that will evaluate its energy.

    Args:
        folders (list[str]): Folders whose energies are combined.
        label (str): Name of the calculation, used in the message.
        model_load_path (str): Model used to evaluate energies that are not read from DFT.

    Returns:
        sources (dict): folder -> source.
    """
    sources = {folder: get_energy_source(folder) for folder in folders}
    families = set(s for s in sources.values() if s is not None)

    if len(families) > 1:
        print(f'[warning] {label}: energies come from different levels of theory ({sorted(families)}); '
              f'the resulting energy differences are not meaningful.')
        for source in sorted(families):
            members = [os.path.basename(f.rstrip('/')) for f, s in sources.items() if s == source]
            print(f'    {source}: {members[:6]}{" ..." if len(members) > 6 else ""}')

    for folder, source in sources.items():
        info = _read_provenance(folder)
        if source == 'ML-IAP' and info is not None and info.get('model') != model_load_path:
            print(f"[warning] {label}: {folder} was relaxed with '{info.get('model')}' "
                  f"but its energy is evaluated with '{model_load_path}'.")
    return sources


def read_energy(
        folder,
        model_load_path=DEFAULT_MODEL,
        device=DEFAULT_DEVICE
):
    """Read the energy of a structure from a given folder.

    Priority: vasprun.xml (DFT) > single_shot_energy file > ML-IAP single point on CONTCAR. If there is
    only a POSCAR, it is first relaxed with the ML-IAP (fixed cell) to produce the CONTCAR. A CONTCAR
    that already exists is trusted as relaxed.

    Args:
        folder (str): Path to the folder containing the structure.
        model_load_path (str): Path to the pre-trained MACE model file. Default is the 'mace-mpa-0-medium.model' model.
        device (str): Device to run the computation on, e.g. 'cuda' or 'cpu'. Default is 'cpu'.

    Returns:
        ssc_energy (float): Single-shot energy of the structure (NaN if it could not be obtained)
    """
    ssc_energy = np.nan
    vasprun_file = None
    for candidate in ['vasprun.xml', 'vasprun.xml.gz']:
        path = os.path.join(folder, candidate)
        if os.path.exists(path):
            vasprun_file = path
            break

    if vasprun_file is not None:
        try:
            ssc_energy = Vasprun(vasprun_file).final_energy
        except Exception as exc:
            print(f'Error reading {os.path.basename(vasprun_file)} at {folder}: {exc}')
    elif os.path.exists(f'{folder}/single_shot_energy'):
        ssc_energy = np.loadtxt(f'{folder}/single_shot_energy')
    elif os.path.exists(f'{folder}/CONTCAR') or os.path.exists(f'{folder}/POSCAR'):
        try:
            structure_file = vasp_files.get_structure_file(folder)
            if structure_file is None:
                raise FileNotFoundError(f'No structure file found in {folder}')
            if not os.path.exists(f'{folder}/CONTCAR'):
                _ = mlip.structural_relaxation(structure_file,
                                              model_load_path,
                                              device=device,
                                              relax_cell=False,
                                              output_folder=folder)

            ssc_energy, _, _ = mlip.single_shot_energy_calculation(f'{folder}/CONTCAR',
                                                                  model_load_path,
                                                                  device=device)
        except Exception as exc:
            print(f'Error loading model or structure in {folder}: {type(exc).__name__}: {exc}')
    return ssc_energy
