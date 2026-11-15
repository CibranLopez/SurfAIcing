"""Layout of a calculation folder, reading VASP-format structures and writing k-point meshes.

Every calculation folder (bulk, slab, adsorption configuration, interface, ...) has the same layout:

    <folder>/POSCAR, KPOINTS, POTCAR, INCAR, run.sh   inputs (those that exist); POSCAR is the structure to relax
    <folder>/slab_data.json                           bulk / slab metadata: only in the root, never copied
    <folder>/relaxation/                              copies of the inputs and what the 0 K calculation produces:
                                                      CONTCAR, run.traj, mlip_provenance.json, vasprun.xml, OUTCAR,
                                                      single_shot_energy (``prepare_relaxation_folder``)
    <folder>/md/<T>K/                                 NPT runs (``libraries.thermal``)
    <folder>/alignment/                               LOCPOT and vasprun.xml of the band alignment and its results

A structure is read from ``relaxation/CONTCAR`` if it exists, otherwise from the POSCAR of the root.
"""
import os
import shutil

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun

RELAXATION_DIR = 'relaxation'
MD_DIR         = 'md'
ALIGNMENT_DIR  = 'alignment'
INPUT_FILES    = ('POSCAR', 'KPOINTS', 'POTCAR', 'INCAR', 'run.sh')   # the files that are copied into the run folders


def relaxation_dir(folder):
    """``<folder>/relaxation``: inputs and results of the 0 K calculation of the structure in `folder`."""
    return os.path.join(folder, RELAXATION_DIR)


def alignment_dir(folder):
    """``<folder>/alignment``: LOCPOT, vasprun.xml and results of the band alignment of the structure in `folder`."""
    return os.path.join(folder, ALIGNMENT_DIR)


def prepare_relaxation_folder(folder):
    """Create ``<folder>/relaxation`` and copy into it the input files that exist in the root of `folder`.

    Only the files of ``INPUT_FILES`` are copied (``slab_data.json`` stays in the root), and a file that is already there is
    kept, so a relaxation that has run is never overwritten.

    Returns:
        str: the path of the relaxation folder.
    """
    target = relaxation_dir(folder)
    os.makedirs(target, exist_ok=True)
    for name in INPUT_FILES:
        source, destination = os.path.join(folder, name), os.path.join(target, name)
        if os.path.exists(source) and not os.path.exists(destination):
            shutil.copy(source, destination)
    return target


def find_vasprun(directory):
    """Path of the vasprun.xml (or vasprun.xml.gz) in `directory`, or None."""
    for candidate in ['vasprun.xml', 'vasprun.xml.gz']:
        path = os.path.join(directory, candidate)
        if os.path.exists(path):
            return path
    return None


def get_structure_file(folder):
    """Return the preferred structure file of a calculation folder.

    ``relaxation/CONTCAR`` (relaxed) has priority over the POSCAR of the root (initial).
    """
    for path in [os.path.join(relaxation_dir(folder), 'CONTCAR'), os.path.join(folder, 'POSCAR')]:
        if os.path.exists(path):
            return path
    return None


def read_structure(folder):
    """Read a structure from the most informative VASP file available in a calculation folder.

    Priority: the vasprun.xml of ``relaxation/`` (final structure), then ``get_structure_file``.
    """
    vasprun_file = find_vasprun(relaxation_dir(folder))
    if vasprun_file is not None:
        try:
            return Vasprun(vasprun_file).final_structure
        except Exception:
            print(f'Error reading {os.path.basename(vasprun_file)} at {relaxation_dir(folder)}')

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
        kpoints_density=(20, 20, 20)
):
    """Generate a KPOINTS file with a given density based on a POSCAR file.

    Args:
        poscar_file     (str):  Path to the POSCAR file.
        kpoints_file    (str):  Path to the KPOINTS file.
        kpoints_density (sequence): K-points density along each reciprocal lattice vector.

    Returns:
        None
    """
    # Read the structure
    structure = Structure.from_file(poscar_file)

    # Calculate the K-points grid based on the given density
    kpoints = Kpoints.automatic_density_by_lengths(structure, kpoints_density)

    # Write the modified KPOINTS file
    kpoints.write_file(kpoints_file)
