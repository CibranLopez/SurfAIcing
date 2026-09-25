import numpy           as np
import libraries.model as slm
import json
import os
import shutil

from scipy.spatial import Voronoi

from pymatgen.io.vasp.inputs import Kpoints, Poscar
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
            'is_symmetric': str(slab.is_symmetric()),
            'name': slab.miller_index
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
        slab_energy,
        bulk_energy_times_fu,
        surface_area
):
    """Compute the surface energy of formation according to:

    \begin{equation}
        E_{surface} = \frac{E_{slab} - E_{bulk}}{2 S}
    \end{equation}

    where all energies are per atom, in units of J/m^2/atom.

    Args:
        ssc_energy_per_atom (float): Single-shot energy per atom.
        bulk_energy_per_atom (float): Bulk energy per atom.
        surface_area (float): Surface area.

    Returns:
        surface_energy_of_formation (float): Surface energy of formation.

    """
    return (slab_energy - bulk_energy_times_fu) * 16.0218 / (2 * surface_area)


def relax_structure(
        poscar_file='POSCAR',
        model_load_path='large',
        relax_cell=False,
        output_folder='.',
        device='cuda'
):
    """Relax a structure.

    Args:
        poscar_file (str): Path to the POSCAR file.
        model_load_path (str): Path to the pre-trained MACE model file. Default is the 'large' model.
        relax_cell (bool): Whether to relax the cell. Default is False.
        output_folder (str): Path to the output folder.
        device (str): Device to run the computation on, e.g. 'cuda' or 'cpu'. Default is 'cuda'.

    Returns:
        None
    """
    # Call MACE relaxer
    try:
        _ = slm.structural_relaxation(poscar_file,
                                      model_load_path,
                                      device=device,
                                      relax_cell=relax_cell,
                                      output_folder=output_folder)
    except:
        print('Error loading model')
        pass


def _get_vasp_structure_file(folder):
    """Return the preferred structure file in a VASP output directory."""
    for candidate in ['CONTCAR', 'POSCAR']:
        path = os.path.join(folder, candidate)
        if os.path.exists(path):
            return path
    return None


def _read_structure_from_folder(folder):
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

    structure_file = _get_vasp_structure_file(folder)
    if structure_file is not None:
        return Structure.from_file(structure_file)

    return None


def read_energy(
        folder,
        model_load_path='mace-mpa-0-medium.model',
        device='cpu'
):
    """Read the energy of a structure from a given folder.

    Args:
        folder (str): Path to the folder containing the structure.
        model_load_path (str): Path to the pre-trained MACE model file. Default is the 'large' model.
        device (str): Device to run the computation on, e.g. 'cuda' or 'cpu'. Default is 'cuda'.

    Returns:
        ssc_energy (float): Single-shot energy of the structure
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
        except Exception:
            print(f'Error reading {os.path.basename(vasprun_file)} at {folder}')
    elif os.path.exists(f'{folder}/single_shot_energy'):
        ssc_energy = np.loadtxt(f'{folder}/single_shot_energy')
    elif os.path.exists(f'{folder}/CONTCAR') or os.path.exists(f'{folder}/POSCAR'):
        try:
            structure_file = _get_vasp_structure_file(folder)
            if structure_file is None:
                raise FileNotFoundError(f'No structure file found in {folder}')
            if not os.path.exists(f'{folder}/CONTCAR'):
                _ = slm.structural_relaxation(structure_file,
                                              model_load_path,
                                              device=device,
                                              relax_cell=False,
                                              output_folder=folder)

            ssc_energy, _, _ = slm.single_shot_energy_calculation(f'{folder}/CONTCAR',
                                                                  model_load_path,
                                                                  device=device)
        except Exception:
            print(f'Error loading model or structure in {folder}')
    return ssc_energy


def read_volume(
        folder
):
    """Read the volume of a structure from a given folder.

    Args:
        folder (str): Path to the folder containing the structure.

    Returns:
        volume (float): Volume of the structure.
    """
    structure = _read_structure_from_folder(folder)
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
        volume (float): Volume of the structure.
    """
    structure = _read_structure_from_folder(folder)
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


def _get_top_surface_sites(
        structure,
        z_tolerance=0.25
):
    """Return the atoms in the top-most surface layer of the slab."""
    if len(structure) == 0:
        raise ValueError('Structure is empty; no surface sites can be sampled.')

    max_z = max(site.z for site in structure)
    top_sites = [site for site in structure if (max_z - site.z) <= z_tolerance]
    if not top_sites:
        top_sites = list(structure)
    return top_sites


def _voronoi_site_groups(
        sites,
        tolerance=1e-6
):
    """Group top-layer sites by their local Voronoi-neighbor geometry.

    This is a lightweight way to approximate inequivalent adsorption positions on a surface
    without assuming that every symmetry-equivalent atom will share the same projected coordinate.
    """
    if len(sites) <= 2:
        return [list(range(len(sites)))]

    coords_2d = np.array([site.coords[:2] for site in sites], dtype=float)
    try:
        vor = Voronoi(coords_2d)
    except Exception:
        return [list(range(len(sites)))]

    neighbor_map = {idx: [] for idx in range(len(sites))}
    for ridge in vor.ridge_points:
        i, j = ridge
        if 0 <= i < len(sites) and 0 <= j < len(sites):
            neighbor_map[int(i)].append(int(j))
            neighbor_map[int(j)].append(int(i))

    local_signatures = []
    for idx, site in enumerate(sites):
        neighbors = neighbor_map.get(idx, [])
        if len(neighbors) == 0:
            local_signatures.append((0.0,))
            continue
        distances = [np.linalg.norm(coords_2d[neighbor] - coords_2d[idx]) for neighbor in neighbors]
        signature = tuple(np.round(np.sort(np.asarray(distances)), decimals=6))
        local_signatures.append(signature)

    groups = []
    for idx, signature in enumerate(local_signatures):
        matched = False
        for group in groups:
            if np.allclose(np.asarray(signature), np.asarray(local_signatures[group[0]]), atol=tolerance, rtol=0):
                group.append(idx)
                matched = True
                break
        if not matched:
            groups.append([idx])

    return groups


def generate_inequivalent_hydrogen_sites(
        surface_dir,
        output_dir=None,
        adsorption_height=1.5,
        z_tolerance=0.25,
        sample_label='conf',
        extra_points_per_site=4,
        offset_fraction=0.35
):
    """Generate a Voronoi-informed set of inequivalent hydrogen adsorption configurations.

    The routine identifies the top-layer surface atoms, groups them via their local Voronoi
    neighbor geometry, and samples a small set of neighboring adsorption points around each unique
    site. This follows the same general idea as the ShakeNBreak defect-sampling approach: collect
    the unique local environments and then generate a few nearby starting points for each one.

    Args:
        surface_dir (str): Directory containing the slab POSCAR (and optional KPOINTS/POTCAR).
        output_dir (str): Directory where the generated adsorption calculations will be written.
            If not provided, a sibling directory called H-absorption is created.
        adsorption_height (float): Height of the H atom above the chosen site in Angstrom.
        z_tolerance (float): Tolerance used to select atoms in the top surface layer.
        sample_label (str): Prefix used for the generated folders.
        extra_points_per_site (int): Number of extra offset points generated around each inequivalent site.
        offset_fraction (float): Fraction of the nearest-neighbor distance used for offset sampling.

    Returns:
        list[str]: Paths to the generated adsorption directories.
    """
    surface_dir = os.path.abspath(surface_dir)
    structure_path = _get_vasp_structure_file(surface_dir)
    if structure_path is None:
        raise FileNotFoundError(f'Neither CONTCAR nor POSCAR found in surface directory: {surface_dir}')

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(surface_dir), 'H-absorption')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    structure = Structure.from_file(structure_path)
    top_sites = _get_top_surface_sites(structure, z_tolerance=z_tolerance)
    if not top_sites:
        raise ValueError(f'Could not identify any top surface sites in {surface_dir}.')

    site_groups = _voronoi_site_groups(top_sites)
    grouped_sites = [top_sites[idx] for group in site_groups for idx in group]
    if not grouped_sites:
        raise ValueError(f'Could not identify any inequivalent top surface sites in {surface_dir}.')

    candidate_centers = []
    for group in site_groups:
        group_sites = [top_sites[idx] for idx in group]
        coords_2d = np.array([site.coords[:2] for site in group_sites], dtype=float)
        center = np.mean(coords_2d, axis=0)
        if len(coords_2d) > 1:
            distances = [np.linalg.norm(coords_2d[i] - center) for i in range(len(coords_2d))]
        else:
            distances = [0.0]
        neighbor_dist = np.median(np.asarray(distances)) if len(distances) > 0 else 0.5
        candidate_centers.append((center, max(neighbor_dist, 0.5)))

    created_dirs = []
    config_index = 0
    for site_index, (center, neighbor_dist) in enumerate(candidate_centers):
        span = max(offset_fraction * neighbor_dist, 0.2)
        offsets = [(0.0, 0.0)]
        for direction in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]:
            offsets.append((direction[0] * span, direction[1] * span))
        for diag in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]:
            if len(offsets) < max(1, extra_points_per_site + 1):
                offsets.append((diag[0] * span / np.sqrt(2), diag[1] * span / np.sqrt(2)))

        offsets = offsets[: max(1, extra_points_per_site + 1)]
        for offset_index, (dx, dy) in enumerate(offsets):
            config_name = f'{sample_label}_{site_index:02d}_{offset_index:02d}'
            config_dir = os.path.join(output_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)
            created_dirs.append(config_dir)

            new_structure = structure.copy()
            adsorption_position = np.array([
                center[0] + dx,
                center[1] + dy,
                max(site.z for site in top_sites) + adsorption_height,
            ], dtype=float)
            new_structure.append('H', adsorption_position, coords_are_cartesian=True)
            Poscar(new_structure).write_file(os.path.join(config_dir, 'POSCAR'))

            for filename in ['KPOINTS', 'POTCAR', 'INCAR', 'run.sh']:
                source = os.path.join(surface_dir, filename)
                if os.path.exists(source):
                    shutil.copy(source, os.path.join(config_dir, filename))

            config_index += 1

    return created_dirs
