"""Bulk preparation, slab metadata and surface energies."""
import json
import os
import shutil

import libraries.energy_lookup as energy_lookup
from libraries.mlip import DEFAULT_DEVICE, DEFAULT_MODEL

from pymatgen.core.structure import Structure


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

    with open(filename, 'w') as json_file:
        json.dump(slab_data, json_file)


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
    r"""Compute the surface energy of formation according to:

    \begin{equation}
        E_{surface} = \frac{E_{slab} - E_{bulk}}{2 S}
    \end{equation}

    The slab energy and the bulk energy (already scaled to the slab composition) are total energies
    in eV and the surface area is in Angstrom^2. The returned value is the surface energy per unit
    area, expressed in J/m^2 (1 eV/Angstrom^2 = 16.0218 J/m^2).

    The factor of 2 assumes that both surfaces of the slab are equivalent. For asymmetric slabs
    the result is the average over the two terminations.

    Args:
        slab_energy (float): Slab energy.
        bulk_energy_times_fu (float): Bulk energy.
        surface_area (float): Surface area.

    Returns:
        surface_energy_of_formation (float): Surface energy of formation in J/m^2.

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
        lines.append(f"    {info['steps']} steps, residual max force (atoms and cell) {info['final_fmax_eV_A']:.3f} eV/Å, "
                     f"residual max |stress| {info['max_abs_stress_GPa']:.3f} GPa")
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
