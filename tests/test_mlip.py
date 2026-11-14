import json

import numpy as np
import pytest

from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read
from ase.io.vasp import read_vasp, write_vasp

import libraries.mlip as mlip


@pytest.fixture
def perturbed_cu(tmp_path):
    """Cubic fcc Cu (4 atoms) with one atom displaced from its lattice site, written as POSCAR."""
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.positions[0] += [0.25, -0.15, 0.1]
    write_vasp(str(tmp_path / 'POSCAR'), atoms, direct=True)
    return tmp_path, atoms


@pytest.fixture
def strained_cu(tmp_path):
    """Cubic fcc Cu with a lattice parameter far from the EMT equilibrium (about 3.6 A)."""
    atoms = bulk('Cu', 'fcc', a=3.9, cubic=True)
    write_vasp(str(tmp_path / 'POSCAR'), atoms, direct=True)
    return tmp_path, atoms


def max_force(atoms):
    atoms.calc = EMT()
    return np.max(np.linalg.norm(atoms.get_forces(), axis=1))


def test_defaults_are_shared_constants():
    assert mlip.DEFAULT_MODEL.endswith('.model')
    assert mlip.DEFAULT_DEVICE == 'cpu'


class TestStructuralRelaxation:
    def test_fixed_cell_relaxation_removes_the_forces(self, perturbed_cu, emt_backend):
        folder, initial = perturbed_cu
        assert max_force(initial.copy()) > 0.05
        relaxed = mlip.structural_relaxation(str(folder / 'POSCAR'), relax_cell=False, output_folder=str(folder))
        assert max_force(relaxed.copy()) < 0.05
        np.testing.assert_allclose(relaxed.cell, initial.cell)       # the cell is untouched

    def test_writes_contcar_trajectory_and_provenance(self, perturbed_cu, emt_backend):
        folder, _ = perturbed_cu
        mlip.structural_relaxation(str(folder / 'POSCAR'), model_load_path='my.model', device='cpu',
                                   relax_cell=False, fmax=0.02, output_folder=str(folder))
        contcar = read_vasp(str(folder / 'CONTCAR'))
        assert max_force(contcar) < 0.02
        assert (folder / 'run.traj').exists()
        info = json.loads((folder / 'mlip_provenance.json').read_text())
        assert info['model'] == 'my.model'
        assert info['relax_cell'] is False
        assert info['fmax'] == 0.02
        assert info['converged'] is True
        assert info['final_fmax_eV_A'] < 0.02
        assert info['max_abs_stress_GPa'] is None                     # no stress without cell relaxation
        assert info['steps'] > 0

    def test_cell_relaxation_reaches_the_equilibrium_lattice(self, strained_cu, emt_backend):
        folder, initial = strained_cu
        relaxed = mlip.structural_relaxation(str(folder / 'POSCAR'), relax_cell=True, output_folder=str(folder))
        assert relaxed.get_volume() < initial.get_volume()
        assert relaxed.cell.lengths()[0] == pytest.approx(3.59, abs=0.05)
        info = json.loads((folder / 'mlip_provenance.json').read_text())
        assert info['relax_cell'] is True
        assert info['max_abs_stress_GPa'] < 0.5                       # residual stress after the relaxation (GPa)
        # the CONTCAR holds the relaxed cell, not the input one
        assert read_vasp(str(folder / 'CONTCAR')).get_volume() == pytest.approx(relaxed.get_volume(), rel=1e-6)

    def test_constant_volume_relaxation_keeps_the_volume(self, strained_cu, emt_backend):
        folder, initial = strained_cu
        relaxed = mlip.structural_relaxation(str(folder / 'POSCAR'), relax_cell=True, constant_volume=True,
                                             output_folder=str(folder))
        assert relaxed.get_volume() == pytest.approx(initial.get_volume(), rel=1e-6)
        assert json.loads((folder / 'mlip_provenance.json').read_text())['constant_volume'] is True

    def test_returns_plain_atoms_also_when_the_cell_is_relaxed(self, strained_cu, emt_backend):
        folder, _ = strained_cu
        relaxed = mlip.structural_relaxation(str(folder / 'POSCAR'), relax_cell=True, output_folder=str(folder))
        assert hasattr(relaxed, 'get_chemical_symbols')               # an Atoms object, not a cell filter

    def test_provenance_records_the_cell_filter(self, strained_cu, emt_backend):
        folder, _ = strained_cu
        mlip.structural_relaxation(str(folder / 'POSCAR'), relax_cell=True, output_folder=str(folder))
        assert json.loads((folder / 'mlip_provenance.json').read_text())['cell_filter'] == 'FrechetCellFilter'

    def test_provenance_has_no_cell_filter_when_the_cell_is_fixed(self, perturbed_cu, emt_backend):
        folder, _ = perturbed_cu
        mlip.structural_relaxation(str(folder / 'POSCAR'), relax_cell=False, output_folder=str(folder))
        assert json.loads((folder / 'mlip_provenance.json').read_text())['cell_filter'] is None

    @pytest.mark.parametrize('constant_volume', [False, True])
    def test_cell_relaxation_uses_the_frechet_cell_filter(self, strained_cu, emt_backend, monkeypatch, constant_volume):
        folder, _ = strained_cu
        used = []

        class Spy(mlip.FrechetCellFilter):
            def __init__(self, atoms, **kwargs):
                used.append(kwargs)
                super().__init__(atoms, **kwargs)

        monkeypatch.setattr(mlip, 'FrechetCellFilter', Spy)
        mlip.structural_relaxation(str(folder / 'POSCAR'), relax_cell=True, constant_volume=constant_volume,
                                   output_folder=str(folder))
        assert used == [{'constant_volume': constant_volume}]
        assert not hasattr(mlip, 'ExpCellFilter')

    def test_no_cell_filter_is_used_when_the_cell_is_fixed(self, perturbed_cu, emt_backend, monkeypatch):
        folder, _ = perturbed_cu

        def must_not_be_called(*args, **kwargs):
            raise AssertionError('a cell filter was created for a fixed-cell relaxation')

        monkeypatch.setattr(mlip, 'FrechetCellFilter', must_not_be_called)
        mlip.structural_relaxation(str(folder / 'POSCAR'), relax_cell=False, output_folder=str(folder))

    def test_cell_forces_are_the_exact_energy_gradient_after_the_cell_has_deformed(self):
        """Why FrechetCellFilter: at a deformed cell its forces on the cell variables equal -dE/dx (finite
        differences), whereas those of the deprecated ExpCellFilter do not."""
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        atoms.positions[0] += [0.12, -0.07, 0.05]
        atoms.calc = EMT()
        cell_filter = mlip.FrechetCellFilter(atoms)
        n = len(atoms)
        x = cell_filter.get_positions().copy()
        x[n:] += 0.3 * np.random.default_rng(1).normal(size=(3, 3))        # deform the cell: log(F) != 0
        cell_filter.set_positions(x)
        forces, x0, h = cell_filter.get_forces().copy(), cell_filter.get_positions().copy(), 1e-5
        for i in range(n, n + 3):
            for j in range(3):
                energies = []
                for step in (+h, -h):
                    y = x0.copy()
                    y[i, j] += step
                    cell_filter.set_positions(y)
                    energies.append(cell_filter.get_potential_energy())
                cell_filter.set_positions(x0)
                assert forces[i, j] == pytest.approx(-(energies[0] - energies[1]) / (2 * h), abs=1e-6)

    def test_unconverged_relaxation_is_reported_in_the_provenance(self, perturbed_cu, emt_backend, monkeypatch):
        folder, _ = perturbed_cu
        import ase.optimize

        class NeverConverges(ase.optimize.BFGS):
            def run(self, fmax=0.05, steps=None):
                return False

        monkeypatch.setattr(mlip, 'BFGS', NeverConverges)
        mlip.structural_relaxation(str(folder / 'POSCAR'), relax_cell=False, output_folder=str(folder))
        assert json.loads((folder / 'mlip_provenance.json').read_text())['converged'] is False

    def test_model_arguments_are_forwarded_to_mace(self, perturbed_cu, emt_backend):
        folder, _ = perturbed_cu
        mlip.structural_relaxation(str(folder / 'POSCAR'), model_load_path='x.model', device='cuda', dispersion=True,
                                   relax_cell=False, output_folder=str(folder))
        assert emt_backend == [{'model': 'x.model', 'device': 'cuda', 'dispersion': True, 'default_dtype': 'float64'}]

    def test_missing_structure_raises(self, tmp_path, emt_backend):
        with pytest.raises(FileNotFoundError):
            mlip.structural_relaxation(str(tmp_path / 'POSCAR'), output_folder=str(tmp_path))


class TestSingleShotEnergy:
    def test_matches_a_direct_evaluation_of_the_potential(self, perturbed_cu, emt_backend):
        folder, atoms = perturbed_cu
        energy, forces, stress = mlip.single_shot_energy_calculation(str(folder / 'POSCAR'))
        reference = read_vasp(str(folder / 'POSCAR'))
        reference.calc = EMT()
        assert energy == pytest.approx(reference.get_potential_energy())
        np.testing.assert_allclose(forces, reference.get_forces())
        np.testing.assert_allclose(stress, reference.get_stress())

    def test_shapes(self, perturbed_cu, emt_backend):
        folder, atoms = perturbed_cu
        energy, forces, stress = mlip.single_shot_energy_calculation(str(folder / 'POSCAR'))
        assert np.isscalar(energy)
        assert forces.shape == (len(atoms), 3)
        assert stress.shape == (6,)

    def test_perfect_crystal_has_no_forces(self, tmp_path, emt_backend):
        write_vasp(str(tmp_path / 'POSCAR'), bulk('Cu', 'fcc', a=3.6, cubic=True), direct=True)
        _, forces, _ = mlip.single_shot_energy_calculation(str(tmp_path / 'POSCAR'))
        np.testing.assert_allclose(forces, 0.0, atol=1e-10)

    def test_energy_is_extensive(self, tmp_path, emt_backend):
        small = bulk('Cu', 'fcc', a=3.6, cubic=True)
        write_vasp(str(tmp_path / 'small'), small, direct=True)
        write_vasp(str(tmp_path / 'large'), small * (2, 1, 1), direct=True)
        e_small = mlip.single_shot_energy_calculation(str(tmp_path / 'small'))[0]
        e_large = mlip.single_shot_energy_calculation(str(tmp_path / 'large'))[0]
        assert e_large == pytest.approx(2 * e_small)

    def test_dispersion_flag_is_forwarded(self, tmp_path, emt_backend):
        write_vasp(str(tmp_path / 'POSCAR'), bulk('Cu', 'fcc', a=3.6, cubic=True), direct=True)
        mlip.single_shot_energy_calculation(str(tmp_path / 'POSCAR'), model_load_path='m', device='cuda', dispersion=True)
        assert emt_backend[-1] == {'model': 'm', 'device': 'cuda', 'dispersion': True, 'default_dtype': 'float64'}


class TestMolecularDynamics:
    @pytest.fixture
    def cu_supercell(self, tmp_path):
        write_vasp(str(tmp_path / 'POSCAR'), bulk('Cu', 'fcc', a=3.6, cubic=True) * (2, 2, 2), direct=True)
        return tmp_path

    def test_runs_and_writes_the_trajectory_and_final_structure(self, cu_supercell, emt_backend):
        initial = read_vasp(str(cu_supercell / 'POSCAR'))
        atoms = mlip.molecular_dynamics(str(cu_supercell / 'POSCAR'), temperature=300, n_steps=20,
                                        output_folder=str(cu_supercell))
        assert len(read(str(cu_supercell / 'run.traj'), index=':')) == 21      # initial frame + 20 steps
        assert (cu_supercell / 'CONTCAR').exists()
        assert len(atoms) == len(initial)
        assert not np.allclose(atoms.positions, initial.positions)            # the atoms moved

    def test_thermal_velocities_are_initialised_at_the_requested_temperature(self, cu_supercell, emt_backend):
        atoms = mlip.molecular_dynamics(str(cu_supercell / 'POSCAR'), temperature=600, n_steps=0,
                                        output_folder=str(cu_supercell))
        assert 200 < atoms.get_temperature() < 1200       # 32 atoms: loose bound around the 600 K target

    def test_zero_kelvin_zero_friction_does_not_heat_a_perfect_crystal(self, cu_supercell, emt_backend):
        atoms = mlip.molecular_dynamics(str(cu_supercell / 'POSCAR'), temperature=0, friction=0.0, n_steps=10,
                                        output_folder=str(cu_supercell))
        assert atoms.get_temperature() < 1.0

    def test_model_arguments_are_forwarded_to_mace(self, cu_supercell, emt_backend):
        mlip.molecular_dynamics(str(cu_supercell / 'POSCAR'), model_load_path='m', device='cuda', n_steps=1,
                                output_folder=str(cu_supercell))
        assert emt_backend[-1]['model'] == 'm' and emt_backend[-1]['device'] == 'cuda'


class TestAseVersionCompatibility:
    """The temperature (kelvin) is passed with the keyword of recent ASE versions, or in eV to old ones."""

    def test_recent_signature_receives_kelvin_as_keyword(self, monkeypatch):
        received = {}

        def recent(atoms, *, temperature_K, comm=None, force_temp=False, rng=None):
            received['temperature_K'] = temperature_K

        monkeypatch.setattr(mlip, 'MaxwellBoltzmannDistribution', recent)
        mlip._initialize_velocities(object(), 300)
        assert received == {'temperature_K': 300}

    def test_old_signature_receives_the_temperature_in_ev(self, monkeypatch):
        received = {}

        def old(atoms, temp, force_temp=False, rng=None):
            received['temp'] = temp

        monkeypatch.setattr(mlip, 'MaxwellBoltzmannDistribution', old)
        mlip._initialize_velocities(object(), 300)
        assert received['temp'] == pytest.approx(300 * units.kB)

    def test_langevin_with_recent_signature(self, monkeypatch):
        received = {}

        class Recent:
            def __init__(self, atoms, timestep, temperature=None, friction=None, fixcm=True, *, temperature_K=None,
                         rng=None, trajectory=None):
                received.update(temperature=temperature, temperature_K=temperature_K, friction=friction)

        monkeypatch.setattr(mlip, 'Langevin', Recent)
        mlip._langevin_dynamics(object(), 1.0, 300, 0.01, trajectory='t.traj')
        assert received == {'temperature': None, 'temperature_K': 300, 'friction': 0.01}

    def test_langevin_with_old_signature(self, monkeypatch):
        received = {}

        class Old:
            def __init__(self, atoms, timestep, temperature=None, friction=None, fixcm=True, trajectory=None):
                received.update(temperature=temperature, friction=friction)

        monkeypatch.setattr(mlip, 'Langevin', Old)
        mlip._langevin_dynamics(object(), 1.0, 300, 0.01, trajectory='t.traj')
        assert received['temperature'] == pytest.approx(300 * units.kB)
