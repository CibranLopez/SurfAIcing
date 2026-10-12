"""Reading VASP-format structures and writing k-point meshes."""
import os

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun


def get_structure_file(folder):
    """Return the preferred structure file in a VASP output directory.

    CONTCAR (relaxed) has priority over POSCAR (initial).
    """
    for candidate in ['CONTCAR', 'POSCAR']:
        path = os.path.join(folder, candidate)
        if os.path.exists(path):
            return path
    return None


def read_structure(folder):
    """Read a structure from the most informative VASP file available in a folder."""
    vasprun_candidates = [
        os.path.join(folder, 'vasprun.xml'),
        os.path.join(folder, 'vasprun.xml.gz'),
    ]
    for vasprun_file in vasprun_candidates:
        if os.path.exists(vasprun_file):
            try:
                return Vasprun(vasprun_file).final_structure
            except Exception:
                print(f'Error reading {os.path.basename(vasprun_file)} at {folder}')
                break

    structure_file = get_structure_file(folder)
    if structure_file is not None:
        return Structure.from_file(structure_file)

    return None


def read_volume(
        folder
):
    """Read the volume of a structure from a given folder.

    Args:
        folder (str): Path to the folder containing the structure.

    Returns:
        volume (float): Volume of the structure.
    """
    structure = read_structure(folder)
    if structure is not None:
        return structure.volume
    print(f'Could not determine structure volume for {folder}')
    return None


def read_lattice_vectors(
        folder
):
    """Read the lattice vectors of a structure from a given folder.

    Args:
        folder (str): Path to the folder containing the structure.

    Returns:
        lattice_vectors (numpy.ndarray): 3x3 matrix with the lattice vectors as rows.
    """
    structure = read_structure(folder)
    if structure is not None:
        return structure.lattice.matrix
    print(f'Could not determine lattice vectors for {folder}')
    return None


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
