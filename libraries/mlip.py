"""ML interatomic potential (MACE) calculations: structural relaxation, single-point energies and molecular dynamics (NVT and NPT).

Relaxations and single points give static (0 K) energies E_0, the zero-temperature level of the free (Gibbs) energy G. The NPT
runs sample G(T, P); ``libraries.thermal`` turns them into the finite-temperature free energy G(T) = U_0(T) + F_vib(T).
"""
import inspect
import json

import numpy as np

from mace.calculators            import mace_mp
from ase.md                      import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase                         import units
from ase.io.vasp                 import read_vasp, write_vasp
from ase.optimize                import BFGS
from ase.filters                 import FrechetCellFilter   # ASE >= 3.23
from scipy.linalg                import rq

try:                                                      # ASE >= 3.23: NPT lives in ase.md.melchionna
    from ase.md.melchionna import MelchionnaNPT as _NPT
except ImportError:                                       # older ASE
    from ase.md.npt import NPT as _NPT

# Defaults shared by every function that evaluates the ML-IAP
DEFAULT_MODEL  = 'mace-mpa-0-medium.model'
DEFAULT_DEVICE = 'cpu'


def _initialize_velocities(atoms, temperature_K, rng=None):
    """Maxwell-Boltzmann velocities at `temperature_K` (kelvin), for old and new ASE versions.

    ASE >= 3.23 takes the temperature in kelvin through the keyword `temperature_K` (the positional energy-unit
    argument was removed in recent releases); older versions only accept the temperature in eV.
    `rng` (numpy Generator) makes the velocities reproducible.
    """
    kwargs = {} if rng is None else {'rng': rng}
    if 'temperature_K' in inspect.signature(MaxwellBoltzmannDistribution).parameters:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, **kwargs)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K * units.kB, **kwargs)


def _langevin_dynamics(atoms, timestep, temperature_K, friction, trajectory):
    """Langevin integrator with the temperature given in kelvin, for old and new ASE versions."""
    if 'temperature_K' in inspect.signature(Langevin.__init__).parameters:
        return Langevin(atoms, timestep=timestep, temperature_K=temperature_K, friction=friction, trajectory=trajectory)
    return Langevin(atoms, timestep=timestep, temperature=temperature_K * units.kB, friction=friction, trajectory=trajectory)


def structural_relaxation(
        path_to_structure,
        model_load_path=DEFAULT_MODEL,
        device=DEFAULT_DEVICE,
        dispersion=False,
        relax_cell=True,
        constant_volume=False,
        fmax=0.05,
        output_folder='./'
):
    """
        Perform structural relaxation of a molecular or crystalline structure.

        This function facilitates the relaxation of a structure file using a pre-trained
        machine learning potential. The relaxation can be constrained to keep the simulation
        cell fixed or allow the relaxation of both positions and the cell itself. The relaxed
        structure is saved to a specified output directory. It is the minimum of the static energy E_0, i.e. the 0 K
        level of the free energy G (at P = 0 also when the cell is relaxed).

        Parameters:
            path_to_structure (str):   Path to the file containing the structure in VASP format.
            model_load_path   (str):   Path to the pre-trained model.
            device            (str):   Device to run the computations on ('cpu' or 'cuda'). Defaults to 'cpu'.
            relax_cell        (bool):  A boolean value indicating whether to relax the simulation cell
                along with atomic positions. Defaults to True. The cell is relaxed with ASE's FrechetCellFilter,
                whose forces on the cell variables are the exact gradient of the energy (ExpCellFilter, used by
                earlier versions of this library, is deprecated and its cell gradients are inconsistent once the
                cell has deformed). The convergence criterion `fmax` applies to the atomic forces and to these
                cell forces together, so the stopping point (and the residual stress, reported in the provenance
                file) can differ slightly from that of an ExpCellFilter relaxation.
            constant_volume   (bool):  A boolean value indicating whether maintain the simulation cell
                volume fixed. Defaults to False.
            fmax              (float): Maximum force tolerance in eV/Å for stopping the relaxation process.
                Defaults to 0.05.

        Returns:
            atoms (Atoms): ASE Atoms object representing the relaxed structure.
    """

    # Load the relaxed structure
    atoms = read_vasp(file=path_to_structure)

    # Load the pre-trained model
    atoms.calc = mace_mp(model=model_load_path, device=device, dispersion=dispersion, default_dtype='float64')

    # Check whether to relax the cell
    if relax_cell:
        atoms = FrechetCellFilter(atoms, constant_volume=constant_volume)

    # Relax the structure
    dyn = BFGS(atoms, trajectory=f'{output_folder}/run.traj')
    converged = dyn.run(fmax=fmax)

    # Residual forces; when the cell is relaxed, `atoms` is the cell filter and these include the cell degrees of freedom
    final_fmax = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))

    if relax_cell:
        atoms = atoms.atoms

    # Residual stress of the relaxed cell in GPa (only meaningful if the cell was relaxed)
    max_abs_stress = float(np.max(np.abs(atoms.get_stress())) * 160.21766208) if relax_cell else None

    write_vasp(f'{output_folder}/CONTCAR', atoms=atoms, direct=True, sort=True)
    
    # Record how the CONTCAR was obtained, so later energy evaluations can check they use the same model
    with open(f'{output_folder}/mlip_provenance.json', 'w') as provenance_file:
        json.dump({'model': model_load_path, 'device': device, 'relax_cell': relax_cell,
                    'constant_volume': constant_volume,
                    'cell_filter': 'FrechetCellFilter' if relax_cell else None, 'fmax': fmax, 'converged': bool(converged),
                    'steps': dyn.get_number_of_steps(), 'final_fmax_eV_A': final_fmax,
                    'max_abs_stress_GPa': max_abs_stress}, provenance_file, indent=2)
    return atoms


def single_shot_energy_calculation(
        path_to_structure,
        model_load_path=DEFAULT_MODEL,
        device=DEFAULT_DEVICE,
        dispersion=False
):
    """
    Determine the energy, forces, and stress of a molecular structure using a pre-trained MACE model.

    This function loads a pre-trained molecular model, reads a relaxed molecular structure from a file,
    and computes the potential energy, atomic forces, and stress tensor for the given molecular configuration.

    Parameters:
        path_to_structure (str):  Path to the file containing the molecular structure
            in VASP format.
        model_load_path   (str):  Path to the pre-trained MACE model file. Defaults to the repository-local 'mace-mpa-0-medium.model'.
        device            (str):  Device to run the computations on, e.g., 'cuda' for GPU or
            'cpu' for CPU. Defaults to 'cpu'.
        dispersion        (bool): Whether to include the D3 dispersion correction in the model.

    Returns:
        float:         The computed potential energy of the structure: its static (0 K) energy E_0, the 0 K level of G.
        numpy.ndarray: The computed forces on every atom in the molecular structure.
        numpy.ndarray: The computed stress tensor of the molecular structure.

    Raises:
        ValueError:   If the molecular structure file is invalid or cannot be read.
        RuntimeError: If the MACE model fails to run the computations due to an
            incompatible model or device.
    """

    # Load the relaxed structure
    atoms = read_vasp(file=path_to_structure)

    # Load the pre-trained model
    atoms.calc = mace_mp(model=model_load_path, device=device, dispersion=dispersion, default_dtype='float64')

    # Determine energy
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    return energy, forces, stress


def molecular_dynamics(
        path_to_structure,
        model_load_path=DEFAULT_MODEL,
        device=DEFAULT_DEVICE,
        dispersion=False,
        temperature=300,
        timestep=1,
        friction=0.001,
        n_steps=200,
        output_folder='./'
):
    """
    Conducts a molecular dynamics simulation on a given atomic structure using a pre-trained model
    and Langevin dynamics for an NVT ensemble. Initializes atomic velocities based on a Maxwell-Boltzmann
    distribution, applies the provided force field, and evolves the system for a specified number of steps.

    Parameters:
        path_to_structure (str):            Path to the input atomic structure file in VASP format.
        model_load_path   (str, optional):  Path or identifier to load the pre-trained model. Defaults to the repository-local 'mace-mpa-0-medium.model'.
        device            (str, optional):  Device used for computation, e.g., 'cuda' or 'cpu'. Defaults to 'cpu'.
        dispersion        (bool, optional): Specifies whether to include dispersion corrections in the model.
            Defaults to False.
        temperature       (float, optional): Initial temperature in Kelvin. Defaults to 300.
        timestep          (float, optional): Timestep for the dynamics in femtoseconds. Defaults to 1.
        friction          (float, optional): Friction coefficient for Langevin dynamics. Defaults to 0.001.
        n_steps           (int, optional):   Number of simulation steps. Defaults to 200.
        output_folder     (str, optional):   Existing folder where run.traj and CONTCAR are written. Defaults to './'.

    Raises:
        Various exceptions may occur during file reading, model initialization, or dynamics execution.
    """

    # Load the relaxed structure
    atoms = read_vasp(file=path_to_structure)

    # Load the pre-trained model
    atoms.calc = mace_mp(model=model_load_path, device=device, dispersion=dispersion, default_dtype='float64')

    # Set units (the temperature stays in kelvin; see _initialize_velocities / _langevin_dynamics)
    timestep *= units.fs
    friction *= 1/units.fs

    # Initialize velocities.
    _initialize_velocities(atoms, temperature)

    # Set up the Langevin dynamics engine for NVT ensemble
    dyn = _langevin_dynamics(atoms, timestep, temperature, friction, trajectory=f'{output_folder}/run.traj')
    dyn.run(n_steps)

    write_vasp(f'{output_folder}/CONTCAR', atoms=atoms, direct=True, sort=True)
    return atoms


def make_cell_upper_triangular(atoms, tol=1e-6, verbose=True):
    """Make the cell of *atoms* upper triangular, as ASE's NPT (Melchionna) requires, in place.

    In ASE the cell is a matrix whose ROWS are the lattice vectors a, b, c.  Upper triangular means
    c_x = c_y = b_x = 0: c along z, b in the yz-plane.  The check in ASE is an exact ``== 0``, so a cell that
    is triangular up to numerical noise (e.g. 3e-15 from a slab built by rotating/transforming a bulk cell) is
    rejected too.

    Cell = (upper triangular) x (orthogonal) is an RQ decomposition: the atoms are rotated rigidly by that
    orthogonal matrix and the fractional coordinates are unchanged, so the structure, its energy and its
    forces are identical; only the orientation in space changes.  If the cell is already triangular within
    *tol* nothing is rotated (the rotation would be the identity) and the noise is just set to exactly zero.

    Call it BEFORE assigning velocities (momenta are not rotated).  Raises ValueError for a left-handed cell
    (det < 0): fixing that needs a mirror, which would invert a chiral crystal.
    """
    cell = np.array(atoms.cell)
    if np.linalg.det(cell) <= 0:
        raise ValueError('left-handed cell (det <= 0): reorder or invert the lattice vectors before running NPT.')

    scaled = atoms.get_scaled_positions(wrap=False)
    lower = np.abs(np.tril(cell, k=-1)).max()                     # b_x, c_x, c_y
    if lower <= tol:
        new_cell = np.triu(cell)                                  # noise -> exact zeros, no rotation
        rotated = False
    else:
        new_cell, q = rq(cell)                                    # cell = new_cell @ q, q orthogonal
        d = np.sign(np.diag(new_cell))                            # positive diagonal -> q is a proper rotation
        new_cell, q = new_cell * d, d[:, None] * q
        new_cell = np.triu(new_cell)
        rotated = True

    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.set_scaled_positions(scaled)                            # same fractional coordinates in the new cell
    if verbose:
        print('Cell made upper triangular' + (' (structure rotated).' if rotated else ' (only numerical noise removed).'))
    return atoms


def npt_dynamics(
        path_to_structure,
        model_load_path=DEFAULT_MODEL,
        device=DEFAULT_DEVICE,
        dispersion=False,
        temperature=300,
        pressure=0.0,
        timestep=1.0,
        ttime=50.0,
        ptime=500.0,
        barostat_modulus=1.0,
        n_steps=20000,
        loginterval=10,
        seed=None,
        output_folder='./'
):
    """
    Constant-pressure, constant-temperature molecular dynamics (NPT) with ASE's Melchionna scheme: a Nose-Hoover
    thermostat combined with a Parrinello-Rahman barostat. The cell is FULLY flexible (no component is frozen), so
    the structure does not need to be relaxed beforehand. The structure may be rotated rigidly (see
    ``make_cell_upper_triangular``); nothing else about it changes.

    NPT samples the Gibbs ensemble G(T, P). The mean potential energy of the run and the vibrational spectrum of its
    momenta give the free energy G(T) = U_0 + F_vib (``libraries.thermal``; no P V term is included there).

    Writes to ``output_folder``: ``output.traj`` (every ``loginterval`` steps: positions, cell, momenta, potential
    energy, forces, stress), ``npt.log`` and ``CONTCAR`` (last frame). The momenta are what the vibrational density of
    states is computed from afterwards (``libraries.thermal``). The total momentum is set to zero, so the system has
    3N - 3 degrees of freedom.

    Parameters:
        path_to_structure (str):   Path to the structure in VASP format.
        model_load_path   (str):   Path to the pre-trained model.
        device            (str):   'cpu' or 'cuda'.
        dispersion        (bool):  Include the D3 correction of the model.
        temperature       (float): Target temperature in kelvin.
        pressure          (float): External pressure in GPa.
        timestep          (float): Time step in femtoseconds.
        ttime             (float): Thermostat time constant in femtoseconds.
        ptime             (float): Barostat time constant in femtoseconds.
        barostat_modulus  (float): Bulk modulus in GPa that sets the barostat mass, pfactor = B * ptime^2 (ASE's recipe).
        n_steps           (int):   Number of MD steps.
        loginterval       (int):   A frame is stored every `loginterval` steps (frame spacing = loginterval * timestep).
        seed              (int):   Seed of the initial velocities (None: random).
        output_folder     (str):   Existing folder for the output files.

    Returns:
        tuple: the final ASE Atoms and a dict with the settings actually used (frame spacing, number of frames, ...).
    """
    atoms = read_vasp(file=path_to_structure)

    # ASE's NPT needs an upper triangular cell (c along z, b in the yz-plane)
    make_cell_upper_triangular(atoms)

    atoms.calc = mace_mp(model=model_load_path, device=device, dispersion=dispersion, default_dtype='float64')

    # Units: fs and GPa -> ASE
    dt_fs = float(timestep)
    timestep_ase = timestep * units.fs
    ttime_ase    = ttime * units.fs
    ptime_ase    = ptime * units.fs
    pressure_ase = pressure / 160.21766208          # GPa -> eV/A^3
    pfactor      = barostat_modulus * units.GPa * ptime_ase ** 2

    rng = None if seed is None else np.random.default_rng(seed)
    _initialize_velocities(atoms, temperature, rng=rng)
    Stationary(atoms)                                # 3N - 3 degrees of freedom

    dyn = _NPT(atoms, timestep=timestep_ase, temperature_K=temperature, ttime=ttime_ase, pfactor=pfactor,
               externalstress=pressure_ase, trajectory=f'{output_folder}/output.traj',
               logfile=f'{output_folder}/npt.log', loginterval=loginterval)
    dyn.run(n_steps)

    write_vasp(f'{output_folder}/CONTCAR', atoms=atoms, direct=True, sort=True)

    info = {'temperature_K': float(temperature), 'pressure_GPa': float(pressure), 'timestep_fs': dt_fs,
            'ttime_fs': float(ttime), 'ptime_fs': float(ptime), 'barostat_modulus_GPa': float(barostat_modulus),
            'n_steps': int(n_steps), 'loginterval': int(loginterval), 'frame_dt_fs': dt_fs * loginterval,
            'n_frames': int(n_steps // loginterval + 1), 'model': model_load_path, 'device': device, 'seed': seed}
    return atoms, info
