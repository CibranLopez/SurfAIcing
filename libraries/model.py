from mace.calculators            import mace_mp
from ase.md                      import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase                         import units
from ase.io.vasp                 import read_vasp, write_vasp
from ase.optimize                import BFGS
from ase.constraints             import ExpCellFilter


linewidth    = 0.5
footnotesize = 8


def structural_relaxation(
        path_to_structure,
        model_load_path='large',
        device='cuda',
        dispersion=False,
        relax_cell=True,
        fmax=0.05,
        output_folder='./'
):
    """
        Perform structural relaxation of a molecular or crystalline structure.

        This function facilitates the relaxation of a structure file using a pre-trained
        machine learning potential. The relaxation can be constrained to keep the simulation
        cell fixed or allow the relaxation of both positions and the cell itself. The relaxed
        structure is saved to a specified output directory.

        Parameters:
            path_to_structure (str):   Path to the file containing the structure in VASP format.
            model_load_path   (str):   Path to the pre-trained model.
            relax_cell        (bool):  A boolean value indicating whether to relax the simulation cell
                along with atomic positions. Defaults to True.
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
        atoms = ExpCellFilter(atoms)

    # Relax the structure
    dyn = BFGS(atoms, trajectory=f'{output_folder}/run.traj')
    dyn.run(fmax=fmax)

    if relax_cell:
        atoms = atoms.atoms

    write_vasp(f'{output_folder}/CONTCAR', atoms=atoms, direct=True, sort=True)
    return atoms


def single_shot_energy_calculation(
        path_to_structure,
        model_load_path='large',
        device='cuda',
        dispersion=False
):
    """
    Determine the energy, forces, and stress of a molecular structure using a pre-trained MACE model.

    This function loads a pre-trained molecular model, reads a relaxed molecular structure from a file,
    and computes the potential energy, atomic forces, and stress tensor for the given molecular configuration.

    Parameters:
        path_to_structure (str):  Path to the file containing the molecular structure
            in VASP format.
        model_load_path   (str):  Path to the pre-trained MACE model file. Default is the 'large' model.
        device            (str):  Device to run the computations on, e.g., 'cuda' for GPU or
            'cpu' for CPU. Default is 'cuda'.
        dispersion        (bool): Whether to include the D3 dispersion correction in the model.

    Returns:
        float:         The computed potential energy of the molecular structure.
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
        model_load_path='large',
        device='cuda',
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
        model_load_path   (str, optional):  Path or identifier to load the pre-trained model. Defaults to 'large'.
        device            (str, optional):  Device used for computation, e.g., 'cuda' or 'cpu'. Defaults to 'cuda'.
        dispersion        (bool, optional): Specifies whether to include dispersion corrections in the model.
            Defaults to False.
        temperature       (float, optional): Initial temperature in Kelvin. Defaults to 300.
        timestep          (float, optional): Timestep for the dynamics in femtoseconds. Defaults to 1.
        friction          (float, optional): Friction coefficient for Langevin dynamics. Defaults to 0.001.
        n_steps           (int, optional):   Number of simulation steps. Defaults to 200.

    Raises:
        Various exceptions may occur during file reading, model initialization, or dynamics execution.
    """

    # Load the relaxed structure
    atoms = read_vasp(file=path_to_structure)

    # Load the pre-trained model
    atoms.calc = mace_mp(model=model_load_path, device=device, dispersion=dispersion, default_dtype='float64')

    # Set units
    timestep    *= units.fs
    temperature *= units.kB
    friction    *= 1/units.fs

    # Initialize velocities.
    MaxwellBoltzmannDistribution(atoms, temperature)

    # Set up the Langevin dynamics engine for NVT ensemble
    dyn = Langevin(atoms, timestep=timestep, temperature=temperature, friction=friction, trajectory=f'{output_folder}/run.traj')
    dyn.run(n_steps)

    write_vasp(f'{output_folder}/CONTCAR', atoms=atoms, direct=True, sort=True)
    return atoms
