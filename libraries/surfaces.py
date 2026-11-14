"""Bulk preparation, slab metadata and surface (free) energies.

Every surface quantity here is a difference of free (Gibbs) energies G of the slab and of the bulk it was cut from. The
functions only do that arithmetic: the G that is passed in decides the level of the result, the static 0 K energy
G = E0 (``energy_lookup.read_energy``) or the finite-temperature G(T) = U0(T) + F_vib(T) (``thermal.thermal_terms``).
"""
import json
import os
import shutil

import numpy as np

import libraries.energy_lookup as energy_lookup
from libraries.mlip import DEFAULT_DEVICE, DEFAULT_MODEL

from pymatgen.core.structure import Structure


def _json_default(value):
    """Let json serialise the NumPy scalars/arrays that pymatgen may leave in the slab metadata."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


# Save slab information
def save_json(
        slab,
        data='slab',
        filename='slab_data.json',
        bulk=None
):
    """Save slab information into json file.

    Args:
        slab (slab): Slab object (or bulk Structure when data='bulk').
        data (str): 'slab' or 'bulk'.
        filename (str): Name of the json file.
        bulk (Structure): Bulk structure the slab was cut from. If given, it is used to flag whether
            the slab is stoichiometric (same reduced formula as the bulk). Without it the flag is 'unknown'.

    Returns:
        None
    """
    if data == 'slab':
        if bulk is None:
            is_stoichiometric = 'unknown'
        else:
            is_stoichiometric = str(slab.composition.reduced_formula == bulk.composition.reduced_formula)

        slab_data = {
            'miller_index': slab.miller_index,
            'shift': slab.shift,
            'surface_area': slab.surface_area,
            'number_of_sites': len(slab.sites),
            'is_polar': str(slab.is_polar()),
            'is_symmetric': str(slab.is_symmetric()),
            'is_stoichiometric': is_stoichiometric,
            'name': slab.miller_index
        }
    elif data == 'bulk':
        slab_data = {
            'number_of_sites': slab.num_sites,
            'reduced_formula': slab.composition.reduced_formula,
        }
    else:
        raise ValueError(f"data must be 'slab' or 'bulk', got {data!r}")

    with open(filename, 'w') as json_file:
        json.dump(slab_data, json_file, default=_json_default)


def load_json(
        filename='slab_data.json'
):
    """Load slab information from json file.

    Args:
        filename (str): Name of the json file.

    Returns:
        slab_data (dict): Dictionary containing the slab information.
    """
    # Load the data from the JSON file
    with open(filename, 'r') as json_file:
        slab_data = json.load(json_file)
    return slab_data


def get_surface_energy_of_formation(
        slab_energy,
        bulk_energy_times_fu,
        surface_area
):
    r"""Surface energy of formation from the free (Gibbs) energies of the slab and of the bulk it was cut from:

    \begin{equation}
        \gamma = \frac{G_{slab} - N g_{bulk}}{2 A}
    \end{equation}

    where ``G_slab`` is the free energy of the slab, ``N g_bulk`` the free energy of the bulk scaled to the composition
    of the slab (N formula units, or the bulk value per atom times the number of atoms of the slab) and ``A`` the area of
    one surface. The function is the same at every level; only the free energies that are passed in change:

    * 0 K (static): G = E_0, the total energy of the relaxed structure (``energy_lookup.read_energy``), and A the 0 K area.
    * finite temperature: G(T) = U_0 + F_vib(T) from NPT runs (``thermal.thermal_terms``), and A the area of the run.

    G is whatever sum of terms the caller includes (today E_0 at 0 K and U_0 + F_vib at finite temperature, with no P V
    term); further terms can be added to G without touching this function. Free energies are in eV and the area in
    Angstrom^2; the result is in J/m^2 (1 eV/Angstrom^2 = 16.0218 J/m^2).

    The factor of 2 assumes that both surfaces of the slab are equivalent. For asymmetric slabs the result is the
    average over the two terminations. A non-stoichiometric slab would need chemical potentials instead of the bulk
    reference.

    Args:
        slab_energy (float): Free energy G_slab of the slab (eV) at the chosen level (E_0 at 0 K).
        bulk_energy_times_fu (float): Free energy of the bulk scaled to the slab composition, N g_bulk (eV), at the same level.
        surface_area (float): Area A of one surface of the slab (Angstrom^2); the area at temperature T for G(T).

    Returns:
        surface_energy_of_formation (float): Surface free energy of formation in J/m^2.

    """
    return (slab_energy - bulk_energy_times_fu) * 16.0218 / (2 * surface_area)


def _relaxation_report(bulk_folder):
    """Describe what the ML-IAP relaxation of the bulk did: lattice before/after and the residual forces and stress."""
    def describe(structure):
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        return (f'a, b, c = {a:.3f}, {b:.3f}, {c:.3f} Å; angles = {alpha:.2f}, {beta:.2f}, {gamma:.2f}°; '
                f'volume = {structure.volume:.3f} Å^3')

    before = Structure.from_file(f'{bulk_folder}/POSCAR')
    after  = Structure.from_file(f'{bulk_folder}/CONTCAR')
    lines  = [f'    input   : {describe(before)}',
              f'    relaxed : {describe(after)}',
              f'    volume change: {100 * (after.volume / before.volume - 1):+.2f}%']

    path = f'{bulk_folder}/mlip_provenance.json'
    if os.path.exists(path):
        with open(path) as json_file:
            info = json.load(json_file)
        stress = info.get('max_abs_stress_GPa')
        stress_text = 'not computed (cell not relaxed)' if stress is None else f'{stress:.3f} GPa'
        lines.append(f"    {info['steps']} steps, residual max force (atoms and cell) {info['final_fmax_eV_A']:.3f} eV/Å, "
                     f"residual max |stress| {stress_text}")
        if not info['converged']:
            lines.append('    [warning] the relaxation did not converge')
    return '\n'.join(lines)


def prepare_bulk_structure(
        structure_file,
        bulk_folder,
        relax=None,
        model_load_path=DEFAULT_MODEL,
        device=DEFAULT_DEVICE
):
    """Prepare the bulk reference folder (POSCAR, CONTCAR) from a user-provided structure file.

    Convention (``relax=None``, inferred from the file name):

    * ``POSCAR*``  -> raw structure (e.g. from a database or from DFT). It is relaxed here with the
      ML-IAP (ions AND cell), so that the bulk energy is the ML-IAP minimum.
    * ``CONTCAR*`` -> structure that was already relaxed somehow. It is used as it is.

    Pass ``relax=True/False`` to override the convention (required for any other file name).

    A structure given as CONTCAR is only consistent if it was relaxed at the same level of theory used
    to evaluate the slabs. A DFT geometry evaluated with the ML-IAP is not at the ML-IAP minimum, which
    biases every surface energy computed from it. The only exceptions are an ML-IAP relaxation done with
    the same model, or a workflow in which all energies (bulk and slabs) come from DFT (vasprun.xml).

    Args:
        structure_file (str): Path to the input structure.
        bulk_folder (str): Folder where POSCAR/CONTCAR of the bulk are written.
        relax (bool): Force (True) or skip (False) the ML-IAP relaxation. None infers it from the file name.
        model_load_path (str): ML-IAP model.
        device (str): Device for the ML-IAP.

    Returns:
        status (str): Human-readable description of what was done.
    """
    name = os.path.basename(structure_file).upper()
    if relax is None:
        if name.startswith('POSCAR'):
            relax = True
        elif name.startswith('CONTCAR'):
            relax = False
        else:
            raise ValueError(f"Cannot infer from the file name '{structure_file}' whether it is already relaxed. "
                             f"Name it POSCAR* (relax with the ML-IAP) or CONTCAR* (already relaxed), or pass relax=True/False.")

    os.makedirs(bulk_folder, exist_ok=True)
    # The input may already be bulk_folder/POSCAR (or CONTCAR): copying a file onto itself raises SameFileError
    if not (os.path.exists(f'{bulk_folder}/POSCAR') and os.path.samefile(structure_file, f'{bulk_folder}/POSCAR')):
        shutil.copy(structure_file, f'{bulk_folder}/POSCAR')

    if relax:
        energy_lookup.relax_structure(f'{bulk_folder}/POSCAR', model_load_path, relax_cell=True, output_folder=bulk_folder, device=device)
        if not os.path.exists(f'{bulk_folder}/CONTCAR'):
            raise RuntimeError(f'The ML-IAP relaxation of {structure_file} failed: no CONTCAR was written to {bulk_folder} '
                               f'(see the error printed above; check the model path and the device).')
        mode   = 'relaxed_with_mlip'
        status = (f'Bulk relaxed with the ML-IAP ({model_load_path}), cell and ions: {structure_file}\n'
                  + _relaxation_report(bulk_folder))
    else:
        if not (os.path.exists(f'{bulk_folder}/CONTCAR') and os.path.samefile(structure_file, f'{bulk_folder}/CONTCAR')):
            shutil.copy(structure_file, f'{bulk_folder}/CONTCAR')
        mode   = 'used_as_already_relaxed'
        status = (f'Bulk used as already relaxed (not verified): {structure_file}. Make sure it was relaxed at the same '
                  f'level of theory used to evaluate the slabs, otherwise surface energies are biased.')
        if name.startswith('POSCAR'):
            status += (f'\n[warning] {os.path.basename(structure_file)} is NOT being relaxed because relax=False was requested; '
                       f'use relax=None (or True) to relax the atoms and the cell with the ML-IAP.')

    with open(f'{bulk_folder}/input_provenance.json', 'w') as json_file:
        json.dump({'input_file': os.path.abspath(structure_file), 'mode': mode,
                   'model': model_load_path if relax else None}, json_file, indent=2)

    return status
