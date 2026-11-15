"""Relaxing structures with the ML-IAP and looking up energies, from DFT (vasprun.xml) or from the ML-IAP.

The energies returned here are static (0 K) energies E_0 of relaxed structures: the zero-temperature level of the free
(Gibbs) energy, G(0 K) = E_0 (no zero-point energy, no entropy, no other thermal term). Finite-temperature free
energies are evaluated from MD runs in ``libraries.thermal``.

The results of the 0 K calculation of a folder live in its ``relaxation/`` subfolder (see ``libraries.vasp_files`` for the
layout). Priority when an energy is requested for a folder: relaxation/vasprun.xml (DFT) > relaxation/single_shot_energy
file > ML-IAP single point on relaxation/CONTCAR (the POSCAR of the folder is first relaxed with the ML-IAP at fixed cell).
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
    mlip.structural_relaxation(poscar_file,
                               model_load_path,
                               device=device,
                               relax_cell=relax_cell,
                               output_folder=output_folder)


def ensure_relaxed(
        folder,
        model_load_path=DEFAULT_MODEL,
        relax_cell=False,
        device=DEFAULT_DEVICE
):
    """Make sure ``<folder>/relaxation/CONTCAR`` exists and return its path.

    The relaxation folder is created with copies of the inputs of `folder` (``vasp_files.prepare_relaxation_folder``). If it
    has no CONTCAR yet, its POSCAR is relaxed with the ML-IAP there (cell fixed unless `relax_cell`); a CONTCAR that already
    exists, from this function or from a DFT run, is trusted as relaxed.

    Args:
        folder (str): Calculation folder with a POSCAR in its root (or a ready relaxation/CONTCAR).
        model_load_path (str): Path to the pre-trained MACE model file.
        relax_cell (bool): Whether to relax the cell too.
        device (str): 'cuda' or 'cpu'.

    Returns:
        str: path of ``<folder>/relaxation/CONTCAR``.
    """
    relax_dir = vasp_files.prepare_relaxation_folder(folder)
    contcar = os.path.join(relax_dir, 'CONTCAR')
    if not os.path.exists(contcar):
        poscar = os.path.join(relax_dir, 'POSCAR')
        if not os.path.exists(poscar):
            raise FileNotFoundError(f'No POSCAR in {folder} (nor relaxation/CONTCAR) to relax')
        relax_structure(poscar, model_load_path, relax_cell=relax_cell, output_folder=relax_dir, device=device)
    return contcar


def _read_provenance(folder):
    """Return the ML-IAP relaxation provenance written by libraries.mlip in ``<folder>/relaxation``, or None."""
    path = os.path.join(vasp_files.relaxation_dir(folder), 'mlip_provenance.json')
    if os.path.exists(path):
        with open(path, 'r') as json_file:
            return json.load(json_file)
    return None


def get_energy_source(folder):
    """Return the origin of the energy that read_energy would use for a folder (looked up in ``<folder>/relaxation``).

    Returns:
        str: 'DFT' (vasprun.xml), 'precomputed' (single_shot_energy file), 'ML-IAP' or None.
    """
    relax_dir = vasp_files.relaxation_dir(folder)
    if vasp_files.find_vasprun(relax_dir) is not None:
        return 'DFT'
    if os.path.exists(f'{relax_dir}/single_shot_energy'):
        return 'precomputed'
    if any(os.path.exists(path) for path in (f'{relax_dir}/CONTCAR', f'{relax_dir}/POSCAR', f'{folder}/POSCAR')):
        return 'ML-IAP'
    return None


def check_energy_sources(
        folders,
        label='calculation',
        model_load_path=DEFAULT_MODEL
):
    """Warn if the energies entering one calculation come from different levels of theory.

    Free-energy differences (surface energy, adsorption energy, adhesion energy) are only meaningful if
    every term is evaluated with the same method and at the same level of the free energy (all static 0 K energies,
    or all free energies at the same temperature). read_energy silently prefers vasprun.xml over the
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
    """Read the static (0 K) energy E_0 of a structure from a given folder: the 0 K level of its free energy G.

    The results of the 0 K calculation are looked up in ``<folder>/relaxation``. Priority: vasprun.xml (DFT) >
    single_shot_energy file > ML-IAP single point on CONTCAR. If there is no CONTCAR, the POSCAR of the folder is first
    relaxed with the ML-IAP (fixed cell) into ``relaxation/`` (``ensure_relaxed``). A CONTCAR that already exists is
    trusted as relaxed.

    Args:
        folder (str): Path to the calculation folder.
        model_load_path (str): Path to the pre-trained MACE model file. Default is the 'mace-mpa-0-medium.model' model.
        device (str): Device to run the computation on, e.g. 'cuda' or 'cpu'. Default is 'cpu'.

    Returns:
        ssc_energy (float): Static energy E_0 of the relaxed structure in eV, G at 0 K without zero-point energy
            (NaN if it could not be obtained)
    """
    ssc_energy = np.nan
    relax_dir = vasp_files.relaxation_dir(folder)
    vasprun_file = vasp_files.find_vasprun(relax_dir)

    if vasprun_file is not None:
        try:
            ssc_energy = Vasprun(vasprun_file).final_energy
        except Exception as exc:
            print(f'Error reading {os.path.basename(vasprun_file)} at {relax_dir}: {exc}')
    elif os.path.exists(f'{relax_dir}/single_shot_energy'):
        try:
            ssc_energy = np.loadtxt(f'{relax_dir}/single_shot_energy').item()
        except Exception as exc:
            print(f'Error reading single_shot_energy at {relax_dir}: {type(exc).__name__}: {exc}')
    elif any(os.path.exists(path) for path in (f'{relax_dir}/CONTCAR', f'{relax_dir}/POSCAR', f'{folder}/POSCAR')):
        try:
            contcar = ensure_relaxed(folder, model_load_path, relax_cell=False, device=device)
            ssc_energy, _, _ = mlip.single_shot_energy_calculation(contcar, model_load_path, device=device)
        except Exception as exc:
            print(f'Error loading model or structure in {folder}: {type(exc).__name__}: {exc}')
    return ssc_energy
