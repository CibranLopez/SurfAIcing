import numpy           as np
import libraries.model as slm
import json
import os

from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core.structure import Structure

# Save slab information
def save_json(
        slab,
        data='slab',
        filename='slab_data.json'
):
    """Save slab information into json file.

    Args:
        slab (slab): Slab object.
        filename (str): Name of the json file.

    Returns:
        None
    """
    if data == 'slab':
        slab_data = {
            'miller_index': slab.miller_index,
            'shift': slab.shift,
            'surface_area': slab.surface_area,
            'number_of_sites': len(slab.sites),
            'is_polar': str(slab.is_polar()),
            'is_symmetric': str(slab.is_symmetric())
        }
    elif data == 'bulk':
        slab_data = {
            'number_of_sites': slab.num_sites,
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
        ssc_energy_per_atom,
        bulk_energy_per_atom,
        surface_area
):
    """Compute the surface energy of formation according to:

    Args:

    Returns:

    """
    return (ssc_energy_per_atom - bulk_energy_per_atom) / (2 * surface_area)


def relax_structure(
        poscar_file='POSCAR',
        model_load_path='large',
        relax_cell=False,
        output_folder='.'
):
    """Relax a structure.

    Args:
        poscar_file (str): Path to the POSCAR file.
        model_load_path (str): Path to the pre-trained MACE model file. Default is the 'large' model.
        relax_cell (bool): Whether to relax the cell. Default is False.
        output_folder (str): Path to the output folder.

    Returns:
        None
    """
    # Call MACE relaxer
    try:
        _ = slm.structural_relaxation(poscar_file,
                                      model_load_path,
                                      relax_cell=relax_cell,
                                      output_folder=output_folder)
    except:
        print('Error loading model')
        pass


def read_energy(
        folder,
        model_load_path='large'
):
    """Read the energy of a structure from a given folder.

    Args:
        folder (str): Path to the folder containing the structure.
        model_load_path (str): Path to the pre-trained MACE model file. Default is the 'large' model.

    Returns:
        ssc_energy (float): Single-shot energy of the structure
    """
    ssc_energy = np.nan
    if os.path.exists(f'{folder}/vasprun.xml'):
        try:
            ssc_energy = Vasprun(f'{folder}/vasprun.xml').final_energy
        except:
            print(f'Error reading vasprun.xml at {folder}')
            pass
    elif os.path.exists(f'{folder}/single_shot_energy'):
        ssc_energy = np.loadtxt(f'{folder}/single_shot_energy')
    elif os.path.exists(f'{folder}/CONTCAR') or os.path.exists(f'{folder}/POSCAR'):
        try:
            if not os.path.exists(f'{folder}/CONTCAR'):
                _ = slm.structural_relaxation(f'{folder}/POSCAR',
                                              model_load_path,
                                              relax_cell=False,
                                              output_folder=folder)

            ssc_energy, _, _ = slm.single_shot_energy_calculation(f'{folder}/CONTCAR',
                                                                  model_load_path)
        except:
            print('Error loading model')
            pass
    return ssc_energy


def generate_kpoints(
        poscar_file='POSCAR',
        kpoints_file='KPOINTS',
        kpoints_density=[20, 20, 20]
):
    """Generate a KPOINTS file with a given density based on a POSCAR file.

    Args:
        poscar_file     (str):  Path to the POSCAR file.
        kpoints_file    (str):  Path to the KPOINTS file.
        kpoints_density (list): K-points density.

    Returns:
        None
    """
    # Read the structure
    structure = Structure.from_file(poscar_file)

    # Calculate the K-points grid based on the given density
    kpoints = Kpoints.automatic_density_by_lengths(structure, kpoints_density)

    # Write the modified KPOINTS file
    kpoints.write_file(kpoints_file)