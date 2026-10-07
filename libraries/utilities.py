import numpy           as np
import libraries.model as slm
import json
import os
import shutil

from scipy.spatial import Voronoi

from pymatgen.io.vasp.inputs import Kpoints, Poscar
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core.structure import Structure

# Defaults shared by every function that evaluates the ML-IAP
DEFAULT_MODEL  = 'mace-mpa-0-medium.model'
DEFAULT_DEVICE = 'cpu'

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
        _ = slm.structural_relaxation(poscar_file,
                                      model_load_path,
                                      device=device,
                                      relax_cell=relax_cell,
                                      output_folder=output_folder)
    except:
        print('Error loading model')
        pass


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
        relax_structure(f'{bulk_folder}/POSCAR', model_load_path, relax_cell=True, output_folder=bulk_folder, device=device)
        mode   = 'relaxed_with_mlip'
        status = f'Bulk relaxed with the ML-IAP ({model_load_path}), cell and ions: {structure_file}'
    else:
        shutil.copy(structure_file, f'{bulk_folder}/CONTCAR')
        mode   = 'used_as_already_relaxed'
        status = (f'Bulk used as already relaxed (not verified): {structure_file}. Make sure it was relaxed at the same '
                  f'level of theory used to evaluate the slabs, otherwise surface energies are biased.')

    with open(f'{bulk_folder}/input_provenance.json', 'w') as json_file:
        json.dump({'input_file': os.path.abspath(structure_file), 'mode': mode,
                   'model': model_load_path if relax else None}, json_file, indent=2)

    return status


def _get_vasp_structure_file(folder):
    """Return the preferred structure file in a VASP output directory.

    CONTCAR (relaxed) has priority over POSCAR (initial).
    """
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


def _read_provenance(folder):
    """Return the ML-IAP relaxation provenance written by libraries.model, or None."""
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
        except Exception as exc:
            print(f'Error loading model or structure in {folder}: {type(exc).__name__}: {exc}')
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
        lattice_vectors (numpy.ndarray): 3x3 matrix with the lattice vectors as rows.
    """
    structure = _read_structure_from_folder(folder)
    if structure is not None:
        return structure.lattice.matrix
    print(f'Could not determine lattice vectors for {folder}')
    return None


def get_hydrogen_pair_distance(
        folder
):
    """Return the H-H distance in a relaxed H2 adsorption configuration.

    Used to check whether an 'H2' configuration is still molecular after relaxation (about 0.75 A)
    or has dissociated into two chemisorbed H atoms (well above 1 A). Returns None if the folder
    does not contain exactly two H atoms (e.g. the slab itself contains hydrogen).

    Args:
        folder (str): Path to the folder containing the relaxed configuration.

    Returns:
        distance (float): H-H distance in Angstrom, taking periodic images into account.
    """
    structure = _read_structure_from_folder(folder)
    if structure is None:
        return None
    h_indices = [i for i, site in enumerate(structure) if site.specie.symbol == 'H']
    if len(h_indices) != 2:
        return None
    return structure.get_distance(h_indices[0], h_indices[1])


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


def _surface_frame(
        structure,
        surface='top'
):
    """Return the slab with its atoms unwrapped along c, the surface normal and an in-plane basis (u, v).

    The normal is the cross product of the first two lattice vectors, oriented along c, so it is exact
    for any Miller index (a PCA of the atomic positions is NOT: for a thick slab with a small in-plane
    cell the direction of smallest variance lies in the surface plane). The slab is unwrapped
    across the periodic boundary by cutting the cell in the middle of the largest gap along c, i.e. the
    vacuum. ``surface='bottom'`` flips the normal to sample the opposite termination.
    """
    if surface not in ('top', 'bottom'):
        raise ValueError(f"surface must be 'top' or 'bottom', got {surface!r}")

    matrix = structure.lattice.matrix
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, matrix[2]) < 0:
        normal = -normal

    frac = structure.frac_coords.copy()
    sorted_c = np.sort(frac[:, 2] % 1.0)
    gaps = np.diff(np.r_[sorted_c, sorted_c[0] + 1.0])
    slab_start = sorted_c[(int(np.argmax(gaps)) + 1) % len(sorted_c)]
    frac[:, 2] = (frac[:, 2] - slab_start) % 1.0
    unwrapped = Structure(structure.lattice, structure.species, frac)

    if surface == 'bottom':
        normal = -normal
    u = matrix[0] / np.linalg.norm(matrix[0])
    v = np.cross(normal, u)
    return unwrapped, normal, u, v


def _environment_signature(structure, position, cutoff):
    """Sorted (atomic number, distance) list of the slab atoms around a point, used as a fingerprint."""
    neighbors = structure.get_sites_in_sphere(position, cutoff)
    return sorted((int(n.specie.Z), float(n.nn_distance)) for n in neighbors)


def _same_environment(signature_a, signature_b, tolerance):
    """Two fingerprints match if they list the same species and distances within a tolerance."""
    if len(signature_a) != len(signature_b):
        return False
    for (z_a, d_a), (z_b, d_b) in zip(signature_a, signature_b):
        if z_a != z_b or abs(d_a - d_b) > tolerance:
            return False
    return True


def voronoi_adsorption_sites(
        structure,
        normal,
        u,
        v,
        z_tolerance=0.25,
        adsorption_height=1.5,
        env_cutoff=4.5,
        env_tolerance=0.05
):
    """Inequivalent adsorption sites of the top layer from a periodic 2D Voronoi decomposition.

    The top-layer atoms (within ``z_tolerance`` of the highest one along the normal) are projected on
    the surface plane and tiled periodically. Their Voronoi diagram defines three kinds of sites:

    * atop   : on each top-layer atom;
    * bridge : midpoint of each pair of atoms that share a Voronoi ridge (nearest neighbours);
    * hollow : each Voronoi vertex (equidistant from three or more atoms).

    Tiling the layer makes the diagram periodic, so atoms at the cell edge have the same neighbours as
    equivalent atoms in the middle of the cell. Sites are then reduced to the symmetry-inequivalent ones
    by comparing the species and distances of the slab atoms around the actual adsorption position
    (within ``env_cutoff``, which must reach the second layer to tell apart e.g. fcc and hcp hollows). Species are included, so Bi and S atoms of the same layer are never merged.

    Returns:
        list[dict]: one entry per inequivalent site with its 'kind' and Cartesian 'position'.
    """
    heights = structure.cart_coords @ normal
    top_idx = np.where(heights >= heights.max() - z_tolerance)[0]

    a, b = structure.lattice.matrix[0], structure.lattice.matrix[1]

    # In-plane (2D) cell. Everything is done in the orthonormal basis (u, v, normal), so the in-plane
    # component of c (c is not perpendicular to the surface for general Miller indices) never matters.
    cell_2d     = np.array([[a @ u, a @ v], [b @ u, b @ v]])
    cell_2d_inv = np.linalg.inv(cell_2d)

    # Wrap the top-layer atoms into the central 2D cell, then tile that cell periodically
    top_2d = np.c_[structure.cart_coords[top_idx] @ u, structure.cart_coords[top_idx] @ v]
    top_fractional = top_2d @ cell_2d_inv
    top_2d = (top_fractional - np.floor(top_fractional)) @ cell_2d

    shifts = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]
    plane, point_heights, is_central = [], [], []
    for (i, j) in shifts:
        for n, k in enumerate(top_idx):
            plane.append(top_2d[n] + i * cell_2d[0] + j * cell_2d[1])
            point_heights.append(heights[k])
            is_central.append((i, j) == (0, 0))
    plane         = np.array(plane)
    point_heights = np.array(point_heights)
    is_central    = np.array(is_central)

    def inside_cell(point_2d):
        fractional = point_2d @ cell_2d_inv
        return np.all(fractional >= -1e-8) and np.all(fractional < 1 - 1e-8)

    candidates = []  # (kind, 2D position, indices of the atoms that define the site)
    for n in np.where(is_central)[0]:
        candidates.append(('atop', plane[n], [n]))

    voronoi = Voronoi(plane)
    for i, j in voronoi.ridge_points:
        midpoint = 0.5 * (plane[i] + plane[j])
        if inside_cell(midpoint):
            candidates.append(('bridge', midpoint, [i, j]))

    for vertex in voronoi.vertices:
        if inside_cell(vertex):
            distances = np.linalg.norm(plane - vertex, axis=1)
            defining = np.where(distances < distances.min() + 1e-3)[0]
            candidates.append(('hollow', vertex, list(defining)))

    kind_order = {'atop': 0, 'bridge': 1, 'hollow': 2}
    candidates.sort(key=lambda candidate: kind_order[candidate[0]])

    sites, signatures = [], []
    for kind, position_2d, defining in candidates:
        height   = point_heights[defining].max() + adsorption_height
        position = position_2d[0] * u + position_2d[1] * v + height * normal
        signature = _environment_signature(structure, position, env_cutoff)
        if any(_same_environment(signature, other, env_tolerance) for other in signatures):
            continue
        signatures.append(signature)
        sites.append({'kind': kind, 'position': position})

    return sites


def generate_inequivalent_hydrogen_sites(
        surface_dir,
        output_dir=None,
        adsorption_height=1.5,
        z_tolerance=0.25,
        sample_label='conf',
        adsorbate='H',
        h2_bond_length=0.74,
        surface='top',
        env_cutoff=4.5,
        min_distance=1.0,
        overwrite=False
):
    """Generate hydrogen adsorption configurations on the inequivalent sites of a surface.

    The slab is read from CONTCAR (relaxed) if it exists, otherwise from POSCAR. The surface normal is
    computed from the lattice, the top layer is decomposed with a periodic Voronoi tessellation into
    atop, bridge and hollow sites, and symmetry-equivalent sites are removed (see
    ``voronoi_adsorption_sites``). One folder is written per site (and per orientation for H2):
    ``<sample_label>_<index>_<kind>[_<orientation>]``, with a POSCAR and a copy of KPOINTS, POTCAR,
    INCAR and run.sh if they exist in the surface folder. A ``sites.json`` summary is also written.

    Args:
        surface_dir (str): Directory containing the slab CONTCAR/POSCAR (and optional KPOINTS/POTCAR).
        output_dir (str): Directory where the generated adsorption calculations will be written.
            If not provided, a sibling directory called H-absorption is created.
        adsorption_height (float): Height of the (lowest) H atom above the top atoms defining the site, in Angstrom.
        z_tolerance (float): Atoms within this distance of the highest atom belong to the top layer.
        sample_label (str): Prefix used for the generated folders.
        adsorbate (str): 'H' for a hydrogen atom or 'H2' for a hydrogen molecule. For 'H2', each site is
            generated with two orientations (perpendicular and parallel to the surface).
        h2_bond_length (float): H-H bond length in Angstrom, used when adsorbate is 'H2'.
        surface (str): 'top' or 'bottom' termination of the slab (they differ for asymmetric slabs).
        env_cutoff (float): Radius in Angstrom used to compare local environments when merging equivalent sites.
        min_distance (float): Configurations with an H atom closer than this to a slab atom are skipped.
        overwrite (bool): If configurations already exist in output_dir they are reused (their paths are
            returned) unless overwrite=True, which deletes them and generates them again.

    Returns:
        list[str]: Paths to the generated adsorption directories.
    """
    if adsorbate not in ('H', 'H2'):
        raise ValueError(f"adsorbate must be 'H' or 'H2', got {adsorbate!r}")

    surface_dir = os.path.abspath(surface_dir)
    structure_path = _get_vasp_structure_file(surface_dir)
    if structure_path is None:
        raise FileNotFoundError(f'Neither CONTCAR nor POSCAR found in surface directory: {surface_dir}')

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(surface_dir), 'H-absorption')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    existing = sorted(os.path.join(output_dir, name) for name in os.listdir(output_dir)
                      if name.startswith(sample_label) and os.path.isdir(os.path.join(output_dir, name)))
    if existing and not overwrite:
        print(f'{len(existing)} configurations already exist in {output_dir}; reusing them (overwrite=True regenerates them).')
        return existing
    for path in existing:
        shutil.rmtree(path)

    structure, normal, u, v = _surface_frame(Structure.from_file(structure_path), surface=surface)
    sites = voronoi_adsorption_sites(structure, normal, u, v,
                                     z_tolerance=z_tolerance,
                                     adsorption_height=adsorption_height,
                                     env_cutoff=env_cutoff)
    if not sites:
        raise ValueError(f'Could not identify any adsorption site in {surface_dir}.')

    if adsorbate == 'H':
        orientations = [('', [np.zeros(3)])]
    else:
        orientations = [
            ('perp', [np.zeros(3), h2_bond_length * normal]),
            ('para', [-0.5 * h2_bond_length * u, 0.5 * h2_bond_length * u]),
        ]

    created_dirs, summary = [], {}
    for site_index, site in enumerate(sites):
        for orientation_name, displacements in orientations:
            positions = [site['position'] + d for d in displacements]

            # Skip configurations in which an H atom overlaps with the slab
            if any(len(structure.get_sites_in_sphere(p, min_distance)) > 0 for p in positions):
                continue

            config_name = f'{sample_label}_{site_index:02d}_{site["kind"]}'
            if orientation_name:
                config_name += f'_{orientation_name}'
            config_dir = os.path.join(output_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)
            created_dirs.append(config_dir)

            new_structure = structure.copy()
            for position in positions:
                new_structure.append('H', position, coords_are_cartesian=True)
            Poscar(new_structure).write_file(os.path.join(config_dir, 'POSCAR'))

            for filename in ['KPOINTS', 'POTCAR', 'INCAR', 'run.sh']:
                source = os.path.join(surface_dir, filename)
                if os.path.exists(source):
                    shutil.copy(source, os.path.join(config_dir, filename))

            summary[config_name] = {'kind': site['kind'], 'orientation': orientation_name or None,
                                    'cartesian': [float(x) for x in site['position']]}

    with open(os.path.join(output_dir, 'sites.json'), 'w') as json_file:
        json.dump({'surface': surface, 'adsorbate': adsorbate, 'adsorption_height': adsorption_height,
                   'configurations': summary}, json_file, indent=2)

    return created_dirs
