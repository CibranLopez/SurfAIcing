import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ase.build import bulk
from ase.io.vasp import read_vasp, write_vasp
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab, generate_all_slabs

import libraries.energy_lookup as energy_lookup
import libraries.surfaces as surfaces


@pytest.fixture
def nacl():
    return Structure.from_spacegroup('Fm-3m', Lattice.cubic(5.64), ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def cu_bulk():
    return Structure.from_spacegroup('Fm-3m', Lattice.cubic(3.6), ['Cu'], [[0, 0, 0]])


def slab_with_composition(bulk_structure, species):
    """A Slab object holding the given species (to control the stoichiometry)."""
    coords = [[0.0, 0.0, 0.1 * (i + 1)] for i in range(len(species))]
    lattice = Lattice.tetragonal(4.0, 20.0)
    return Slab(lattice, species, coords, miller_index=(1, 1, 1), oriented_unit_cell=bulk_structure,
                shift=0.0, scale_factor=np.eye(3), reorient_lattice=False)


class TestSurfaceEnergy:
    def test_formula_and_unit_conversion(self):
        # (E_slab - E_bulk) / (2 S), with 1 eV/A^2 = 16.0218 J/m^2
        assert surfaces.get_surface_energy_of_formation(-9.9, -10.0, 5.0) == pytest.approx(0.1 / 10.0 * 16.0218)

    def test_unit_conversion_matches_the_physical_constants(self):
        eV, angstrom = 1.602176634e-19, 1e-10
        assert surfaces.get_surface_energy_of_formation(1.0, 0.0, 0.5) == pytest.approx(eV / angstrom ** 2, rel=1e-5)

    def test_zero_when_slab_and_bulk_have_the_same_energy(self):
        assert surfaces.get_surface_energy_of_formation(-10.0, -10.0, 7.0) == 0.0

    def test_sign_follows_the_energy_difference(self):
        assert surfaces.get_surface_energy_of_formation(-9.0, -10.0, 7.0) > 0
        assert surfaces.get_surface_energy_of_formation(-11.0, -10.0, 7.0) < 0

    def test_inversely_proportional_to_the_area(self):
        small = surfaces.get_surface_energy_of_formation(-9.0, -10.0, 4.0)
        large = surfaces.get_surface_energy_of_formation(-9.0, -10.0, 8.0)
        assert small == pytest.approx(2 * large)

    def test_accepts_arrays(self):
        result = surfaces.get_surface_energy_of_formation(np.array([-9.0, np.nan]), -10.0, 5.0)
        assert result[0] > 0 and np.isnan(result[1])

    def test_surface_energy_does_not_depend_on_the_slab_thickness(self):
        """With E_slab = N e_bulk + 2 S gamma, doubling the thickness gives back the same gamma."""
        e_surface_per_area = 0.05                                   # eV/A^2 per surface
        area, e_bulk_per_atom = 10.0, -3.0
        values = []
        for n_atoms in (6, 12):
            slab_energy = n_atoms * e_bulk_per_atom + 2 * area * e_surface_per_area
            values.append(surfaces.get_surface_energy_of_formation(slab_energy, n_atoms * e_bulk_per_atom, area))
        assert values[0] == pytest.approx(values[1])
        assert values[0] == pytest.approx(e_surface_per_area * 16.0218)


class TestSaveAndLoadJson:
    def test_slab_roundtrip(self, cu_bulk, tmp_path):
        slab = generate_all_slabs(cu_bulk, 1, 6, 10, max_normal_search=1)[0]
        path = str(tmp_path / 'slab_data.json')
        surfaces.save_json(slab, filename=path, bulk=cu_bulk)
        data = surfaces.load_json(path)
        assert data['miller_index'] == list(slab.miller_index)
        assert data['name'] == list(slab.miller_index)
        assert data['shift'] == pytest.approx(slab.shift)
        assert data['surface_area'] == pytest.approx(slab.surface_area)
        assert data['number_of_sites'] == len(slab.sites)
        assert data['is_polar'] == 'False'
        assert data['is_symmetric'] == 'True'
        assert data['is_stoichiometric'] == 'True'

    def test_stoichiometry_is_unknown_without_the_bulk(self, cu_bulk, tmp_path):
        slab = generate_all_slabs(cu_bulk, 1, 6, 10, max_normal_search=1)[0]
        surfaces.save_json(slab, filename=str(tmp_path / 'a.json'))
        assert surfaces.load_json(str(tmp_path / 'a.json'))['is_stoichiometric'] == 'unknown'

    def test_non_stoichiometric_slab_is_flagged(self, nacl, tmp_path):
        slab = slab_with_composition(nacl, ['Na', 'Na', 'Na', 'Cl', 'Cl'])
        surfaces.save_json(slab, filename=str(tmp_path / 'a.json'), bulk=nacl)
        assert surfaces.load_json(str(tmp_path / 'a.json'))['is_stoichiometric'] == 'False'

    def test_stoichiometric_slab_with_a_different_number_of_formula_units(self, nacl, tmp_path):
        slab = slab_with_composition(nacl, ['Na', 'Cl'] * 3)
        surfaces.save_json(slab, filename=str(tmp_path / 'a.json'), bulk=nacl)
        assert surfaces.load_json(str(tmp_path / 'a.json'))['is_stoichiometric'] == 'True'

    def test_bulk_data(self, nacl, tmp_path):
        surfaces.save_json(nacl, data='bulk', filename=str(tmp_path / 'bulk.json'))
        assert surfaces.load_json(str(tmp_path / 'bulk.json')) == {'number_of_sites': 8, 'reduced_formula': 'NaCl'}

    def test_numpy_values_are_serialised(self, tmp_path):
        """Regression: np.int64 Miller indices (or areas) made json.dump fail."""
        fake_slab = SimpleNamespace(
            miller_index=(np.int64(1), np.int64(1), np.int64(0)), shift=np.float32(0.25), surface_area=np.float64(12.5),
            sites=[0, 1, 2], is_polar=lambda: False, is_symmetric=lambda: True,
            composition=SimpleNamespace(reduced_formula='Cu'))
        surfaces.save_json(fake_slab, filename=str(tmp_path / 'a.json'))
        data = surfaces.load_json(str(tmp_path / 'a.json'))
        assert data['miller_index'] == [1, 1, 0]
        assert data['surface_area'] == 12.5

    def test_numpy_arrays_are_serialised_and_unknown_objects_are_rejected(self):
        assert surfaces._json_default(np.array([1, 2])) == [1, 2]
        assert surfaces._json_default(np.int64(3)) == 3 and type(surfaces._json_default(np.int64(3))) is int
        with pytest.raises(TypeError, match='not JSON serializable'):
            surfaces._json_default(object())

    def test_unknown_data_kind_raises_a_clear_error(self, cu_bulk, tmp_path):
        """Regression: an UnboundLocalError was raised instead."""
        with pytest.raises(ValueError, match="data must be 'slab' or 'bulk'"):
            surfaces.save_json(cu_bulk, data='surface', filename=str(tmp_path / 'a.json'))
        assert not (tmp_path / 'a.json').exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            surfaces.load_json(str(tmp_path / 'missing.json'))


class TestPrepareBulkStructure:
    @pytest.fixture
    def strained_poscar(self, tmp_path):
        path = tmp_path / 'input' / 'POSCAR_Cu'
        path.parent.mkdir()
        write_vasp(str(path), bulk('Cu', 'fcc', a=3.9, cubic=True), direct=True)
        return path

    def test_poscar_is_relaxed_with_the_mliap(self, strained_poscar, tmp_path, emt_backend):
        bulk_folder = tmp_path / 'bulk'
        status = surfaces.prepare_bulk_structure(str(strained_poscar), str(bulk_folder))
        assert read_vasp(str(bulk_folder / 'CONTCAR')).get_volume() < read_vasp(str(bulk_folder / 'POSCAR')).get_volume()
        assert 'Bulk relaxed with the ML-IAP' in status
        assert 'volume change' in status and 'residual max |stress|' in status
        provenance = json.loads((bulk_folder / 'input_provenance.json').read_text())
        assert provenance['mode'] == 'relaxed_with_mlip'
        assert provenance['input_file'] == str(strained_poscar)
        assert (bulk_folder / 'mlip_provenance.json').exists()

    def test_the_cell_is_relaxed_too(self, strained_poscar, tmp_path, emt_backend):
        surfaces.prepare_bulk_structure(str(strained_poscar), str(tmp_path / 'bulk'))
        assert json.loads((tmp_path / 'bulk' / 'mlip_provenance.json').read_text())['relax_cell'] is True

    def test_contcar_is_used_as_it_is(self, tmp_path, emt_backend):
        source = tmp_path / 'CONTCAR_Cu'
        write_vasp(str(source), bulk('Cu', 'fcc', a=3.9, cubic=True), direct=True)
        status = surfaces.prepare_bulk_structure(str(source), str(tmp_path / 'bulk'))
        assert 'used as already relaxed' in status and 'not verified' in status
        assert (tmp_path / 'bulk' / 'CONTCAR').read_text() == source.read_text()
        assert (tmp_path / 'bulk' / 'POSCAR').read_text() == source.read_text()
        assert emt_backend == []                                    # no relaxation was run
        assert json.loads((tmp_path / 'bulk' / 'input_provenance.json').read_text()) == {
            'input_file': str(source), 'mode': 'used_as_already_relaxed', 'model': None}

    def test_relax_can_be_forced_for_a_contcar(self, tmp_path, emt_backend):
        source = tmp_path / 'CONTCAR_Cu'
        write_vasp(str(source), bulk('Cu', 'fcc', a=3.9, cubic=True), direct=True)
        surfaces.prepare_bulk_structure(str(source), str(tmp_path / 'bulk'), relax=True)
        assert read_vasp(str(tmp_path / 'bulk' / 'CONTCAR')).get_volume() < 3.9 ** 3 - 1

    def test_relax_false_on_a_poscar_warns(self, strained_poscar, tmp_path, emt_backend):
        status = surfaces.prepare_bulk_structure(str(strained_poscar), str(tmp_path / 'bulk'), relax=False)
        assert '[warning]' in status and 'NOT being relaxed' in status
        assert emt_backend == []

    def test_other_file_names_need_an_explicit_choice(self, tmp_path, emt_backend):
        source = tmp_path / 'structure.vasp'
        write_vasp(str(source), bulk('Cu', 'fcc', a=3.9, cubic=True), direct=True)
        with pytest.raises(ValueError, match='Cannot infer'):
            surfaces.prepare_bulk_structure(str(source), str(tmp_path / 'bulk'))
        surfaces.prepare_bulk_structure(str(source), str(tmp_path / 'bulk'), relax=False)
        assert (tmp_path / 'bulk' / 'CONTCAR').exists()

    def test_file_name_check_is_case_insensitive(self, tmp_path, emt_backend):
        source = tmp_path / 'poscar_cu'
        write_vasp(str(source), bulk('Cu', 'fcc', a=3.9, cubic=True), direct=True)
        surfaces.prepare_bulk_structure(str(source), str(tmp_path / 'bulk'))
        assert emt_backend                                          # it was relaxed

    def test_a_failed_relaxation_is_not_hidden(self, strained_poscar, tmp_path, broken_backend):
        with pytest.raises(RuntimeError, match='cannot load the model'):
            surfaces.prepare_bulk_structure(str(strained_poscar), str(tmp_path / 'bulk'))

    def test_a_relaxation_that_writes_nothing_raises(self, strained_poscar, tmp_path, monkeypatch):
        monkeypatch.setattr(energy_lookup, 'relax_structure', lambda *args, **kwargs: None)
        with pytest.raises(RuntimeError, match='no CONTCAR was written'):
            surfaces.prepare_bulk_structure(str(strained_poscar), str(tmp_path / 'bulk'))

    def test_input_already_inside_the_bulk_folder(self, tmp_path, emt_backend):
        """Regression: copying the file onto itself raised SameFileError."""
        bulk_folder = tmp_path / 'bulk'
        bulk_folder.mkdir()
        write_vasp(str(bulk_folder / 'POSCAR'), bulk('Cu', 'fcc', a=3.9, cubic=True), direct=True)
        surfaces.prepare_bulk_structure(str(bulk_folder / 'POSCAR'), str(bulk_folder))
        assert (bulk_folder / 'CONTCAR').exists()

    def test_contcar_already_inside_the_bulk_folder(self, tmp_path):
        bulk_folder = tmp_path / 'bulk'
        bulk_folder.mkdir()
        write_vasp(str(bulk_folder / 'CONTCAR'), bulk('Cu', 'fcc', a=3.9, cubic=True), direct=True)
        surfaces.prepare_bulk_structure(str(bulk_folder / 'CONTCAR'), str(bulk_folder))
        assert (bulk_folder / 'POSCAR').read_text() == (bulk_folder / 'CONTCAR').read_text()


class TestRelaxationReport:
    @staticmethod
    def make_folder(tmp_path, provenance=None):
        folder = tmp_path / 'bulk'
        folder.mkdir()
        write_vasp(str(folder / 'POSCAR'), bulk('Cu', 'fcc', a=3.9, cubic=True), direct=True)
        write_vasp(str(folder / 'CONTCAR'), bulk('Cu', 'fcc', a=3.6, cubic=True), direct=True)
        if provenance is not None:
            (folder / 'mlip_provenance.json').write_text(json.dumps(provenance))
        return str(folder)

    def test_lattice_and_volume_change(self, tmp_path):
        report = surfaces._relaxation_report(self.make_folder(tmp_path))
        assert 'a, b, c = 3.900, 3.900, 3.900' in report
        assert 'a, b, c = 3.600, 3.600, 3.600' in report
        assert f'{100 * ((3.6 / 3.9) ** 3 - 1):+.2f}%' in report

    def test_includes_the_provenance_when_available(self, tmp_path):
        info = {'steps': 7, 'final_fmax_eV_A': 0.012, 'max_abs_stress_GPa': 0.034, 'converged': True}
        report = surfaces._relaxation_report(self.make_folder(tmp_path, info))
        assert '7 steps' in report and '0.012 eV/Å' in report and '0.034 GPa' in report
        assert 'did not converge' not in report

    def test_unconverged_relaxation_is_flagged(self, tmp_path):
        info = {'steps': 7, 'final_fmax_eV_A': 0.2, 'max_abs_stress_GPa': 0.5, 'converged': False}
        assert '[warning] the relaxation did not converge' in surfaces._relaxation_report(self.make_folder(tmp_path, info))

    def test_missing_stress_does_not_break_the_report(self, tmp_path):
        info = {'steps': 3, 'final_fmax_eV_A': 0.01, 'max_abs_stress_GPa': None, 'converged': True}
        assert 'not computed' in surfaces._relaxation_report(self.make_folder(tmp_path, info))
