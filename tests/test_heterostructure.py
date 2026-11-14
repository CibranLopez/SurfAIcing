import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ase.io.vasp import read_vasp
from pymatgen.core import Lattice, Structure
from pymatgen.core.interface import Interface
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.core.surface import Slab

import libraries.heterostructure as heterostructure
import libraries.mlip as mlip
from tests.helpers import hex_lattice, oblique_slab, rotate_about_z, two_layer_hex_slab


def area_of(structure):
    return float(np.linalg.norm(np.cross(*structure.lattice.matrix[:2])))


def periodic_difference(first, second):
    """Fractional difference reduced to [-0.5, 0.5)."""
    difference = np.asarray(first) - np.asarray(second)
    return difference - np.round(difference)


@pytest.fixture(scope='module')
def substrate():
    return heterostructure.wrap_as_slab(two_layer_hex_slab(3.0, ('Ti', 'O')), (0, 0, 1))


@pytest.fixture(scope='module')
def film():
    return heterostructure.wrap_as_slab(two_layer_hex_slab(3.0 * 1.01, ('Zn', 'S')), (1, 1, 1))


@pytest.fixture(scope='module')
def generator():
    return heterostructure.make_zsl_generator(100, 0.09, 0.03, 0.03)


@pytest.fixture(scope='module')
def matches(film, substrate, generator):
    return heterostructure.find_matches(film, substrate, generator, 5)


@pytest.fixture(scope='module')
def supercells(film, substrate, matches):
    return heterostructure.build_supercell_slabs(film, substrate, matches[0])


# --------------------------------------------------------------------------------------------------------------
# Reading the slabs
# --------------------------------------------------------------------------------------------------------------
class TestLargestGap:
    def test_gap_between_the_slab_and_its_periodic_image(self):
        structure = Structure(Lattice.tetragonal(3.0, 20.0), ['Cu', 'Cu'], [[0, 0, 0.1], [0, 0, 0.3]])
        assert heterostructure._largest_gap(structure, 2) == pytest.approx(0.8)

    def test_slab_crossing_the_boundary(self):
        structure = Structure(Lattice.tetragonal(3.0, 20.0), ['Cu', 'Cu'], [[0, 0, 0.9], [0, 0, 0.1]])
        assert heterostructure._largest_gap(structure, 2) == pytest.approx(0.8)

    def test_equally_spaced_atoms(self):
        structure = Structure(Lattice.cubic(4.0), ['Cu'] * 4, [[0, 0, i / 4] for i in range(4)])
        assert heterostructure._largest_gap(structure, 2) == pytest.approx(0.25)

    def test_each_axis_is_independent(self):
        structure = Structure(Lattice.cubic(10.0), ['Cu', 'Cu'], [[0.0, 0.0, 0.0], [0.1, 0.5, 0.0]])
        assert heterostructure._largest_gap(structure, 0) == pytest.approx(0.9)
        assert heterostructure._largest_gap(structure, 1) == pytest.approx(0.5)


class TestCheckSlabFile:
    def test_a_good_slab_reports_area_and_vacuum_without_warnings(self, tmp_path, capsys):
        slab = two_layer_hex_slab(3.0, vacuum=15.0, spacing=2.0)
        heterostructure.check_slab_file(slab, 'substrate', str(tmp_path / 'POSCAR'))
        out = capsys.readouterr().out
        assert 'substrate: in-plane area 7.8 Å², vacuum along the normal 17.0 Å' in out
        assert '[warning]' not in out

    def test_bulk_like_cell_is_flagged(self, tmp_path, capsys):
        slab = two_layer_hex_slab(3.0, vacuum=1.0, spacing=2.0)
        heterostructure.check_slab_file(slab, 'film', str(tmp_path / 'POSCAR'))
        out = capsys.readouterr().out
        assert "[warning] 'film' has only 3.0 Å of vacuum along c" in out

    def test_vacuum_along_an_in_plane_axis_is_flagged(self, tmp_path, capsys):
        structure = Structure(Lattice([[30, 0, 0], [0, 3, 0], [0, 0, 3]]), ['Cu', 'Cu'], [[0, 0, 0], [0.05, 0.5, 0.5]])
        heterostructure.check_slab_file(structure, 'substrate', str(tmp_path / 'POSCAR'))
        out = capsys.readouterr().out
        assert "'substrate' has 28.5 Å of vacuum along a" in out
        assert 'vacuum along b' not in out

    def test_in_plane_vacuum_is_measured_perpendicular_to_the_other_two_axes(self, tmp_path, capsys):
        """A sheared cell: the heights are cell-volume / face-area, not the lattice parameters."""
        structure = Structure(Lattice([[3, 0, 0], [3, 20, 0], [0, 0, 20]]), ['Cu', 'Cu'], [[0, 0, 0], [0.1, 0.1, 0.0]])
        heterostructure.check_slab_file(structure, 'film', str(tmp_path / 'POSCAR'))
        assert 'vacuum along b' in capsys.readouterr().out

    def write_json(self, folder, **data):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, 'slab_data.json'), 'w') as handle:
            json.dump(data, handle)
        return os.path.join(folder, 'POSCAR')

    def test_matching_slab_data_is_silent(self, tmp_path, capsys):
        slab = two_layer_hex_slab()
        path = self.write_json(str(tmp_path), surface_area=area_of(slab), miller_index=[0, 0, 1])
        heterostructure.check_slab_file(slab, 'substrate', path, expected_miller=(0, 0, 1))
        assert '[warning]' not in capsys.readouterr().out

    def test_area_that_differs_from_the_stored_one_is_flagged(self, tmp_path, capsys):
        slab = two_layer_hex_slab()
        path = self.write_json(str(tmp_path), surface_area=2 * area_of(slab), miller_index=[0, 0, 1])
        heterostructure.check_slab_file(slab, 'film', path)
        assert 'differs from the area stored in slab_data.json' in capsys.readouterr().out

    def test_small_area_differences_are_tolerated(self, tmp_path, capsys):
        slab = two_layer_hex_slab()
        path = self.write_json(str(tmp_path), surface_area=area_of(slab) * 1.005)
        heterostructure.check_slab_file(slab, 'film', path)
        assert '[warning]' not in capsys.readouterr().out

    def test_wrong_miller_index_is_flagged(self, tmp_path, capsys):
        slab = two_layer_hex_slab()
        path = self.write_json(str(tmp_path), surface_area=area_of(slab), miller_index=[1, 1, 1])
        heterostructure.check_slab_file(slab, 'film', path, expected_miller=(0, 0, 1))
        out = capsys.readouterr().out
        assert 'Miller index (1, 1, 1), but the declared Miller index is (0, 0, 1)' in out

    def test_miller_index_is_not_checked_unless_declared(self, tmp_path, capsys):
        slab = two_layer_hex_slab()
        path = self.write_json(str(tmp_path), surface_area=area_of(slab), miller_index=[1, 1, 1])
        heterostructure.check_slab_file(slab, 'film', path)
        assert '[warning]' not in capsys.readouterr().out

    def test_bulk_slab_data_without_slab_keys_does_not_crash(self, tmp_path, capsys):
        """Regression: pointing at a bulk slab_data.json (no surface_area / miller_index) raised KeyError."""
        path = self.write_json(str(tmp_path), number_of_sites=4, reduced_formula='Cu')
        heterostructure.check_slab_file(two_layer_hex_slab(), 'film', path, expected_miller=(1, 1, 1))
        assert 'in-plane area' in capsys.readouterr().out

    def test_missing_json_is_fine(self, tmp_path, capsys):
        heterostructure.check_slab_file(two_layer_hex_slab(), 'film', str(tmp_path / 'POSCAR'), expected_miller=(1, 1, 1))
        assert '[warning]' not in capsys.readouterr().out


class TestWrapAsSlab:
    def test_returns_a_slab_with_the_same_atoms_and_cell(self):
        structure = two_layer_hex_slab()
        slab = heterostructure.wrap_as_slab(structure, (1, 1, 1))
        assert isinstance(slab, Slab)
        assert slab.miller_index == (1, 1, 1)
        assert slab.lattice == structure.lattice                    # the lattice is not re-oriented
        assert [s.symbol for s in slab.species] == [s.symbol for s in structure.species]
        np.testing.assert_allclose(slab.frac_coords, structure.frac_coords)

    def test_lattice_is_not_reoriented_for_an_arbitrarily_oriented_cell(self):
        structure = rotate_about_z(two_layer_hex_slab(), 37.0)
        slab = heterostructure.wrap_as_slab(structure, (0, 0, 1))
        np.testing.assert_allclose(slab.lattice.matrix, structure.lattice.matrix)

    def test_supercells_and_copies_work(self):
        slab = heterostructure.wrap_as_slab(two_layer_hex_slab(), (0, 0, 1))
        copy = slab.copy()
        copy.make_supercell([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert len(copy) == 4 and len(slab) == 2


def test_describe_lattice():
    slab = heterostructure.wrap_as_slab(two_layer_hex_slab(3.0), (0, 0, 1))
    assert heterostructure.describe_lattice(slab) == '|a| = 3.00 Å, |b| = 3.00 Å, angle = 120.0°, area = 7.8 Å²'


def test_describe_lattice_oblique():
    slab = heterostructure.wrap_as_slab(oblique_slab(a=3.0, b=4.0, gamma=75.0), (0, 0, 1))
    text = heterostructure.describe_lattice(slab)
    assert '|a| = 3.00 Å, |b| = 4.00 Å, angle = 75.0°' in text
    assert f'area = {3.0 * 4.0 * np.sin(np.radians(75.0)):.1f} Å²' in text


# --------------------------------------------------------------------------------------------------------------
# Lattice matching
# --------------------------------------------------------------------------------------------------------------
class TestZslGenerator:
    def test_settings_are_forwarded(self):
        generator = heterostructure.make_zsl_generator(120, 0.07, 0.02, 0.015)
        assert generator.max_area == 120
        assert generator.max_area_ratio_tol == 0.07
        assert generator.max_length_tol == 0.02
        assert generator.max_angle_tol == 0.015
        assert generator.bidirectional is True


class TestMatchStrain:
    def test_identical_lattices_have_no_strain(self, substrate, generator):
        matches = heterostructure.find_matches(substrate, substrate, generator)
        assert heterostructure.match_strain(matches[0]) == pytest.approx(0.0, abs=1e-9)

    def test_strain_grows_with_the_mismatch(self, substrate):
        generator = heterostructure.make_zsl_generator(100, 0.09, 0.06, 0.06)
        strains = []
        for scale in (1.01, 1.03, 1.05):
            film = heterostructure.wrap_as_slab(two_layer_hex_slab(3.0 * scale), (1, 1, 1))
            strains.append(heterostructure.match_strain(heterostructure.find_matches(film, substrate, generator)[0]))
        assert 0 < strains[0] < strains[1] < strains[2]

    def test_isotropic_strain_value(self, substrate):
        """A film 3 % larger than the substrate is compressed in plane by 1/1.03: von Mises strain of the
        Green-Lagrange tensor 0.5 (F^T F - 1) of that deformation."""
        generator = heterostructure.make_zsl_generator(100, 0.09, 0.05, 0.05)
        film = heterostructure.wrap_as_slab(two_layer_hex_slab(3.0 * 1.03), (1, 1, 1))
        best = heterostructure.find_matches(film, substrate, generator)[0]
        deformation = Deformation(np.diag([1 / 1.03, 1 / 1.03, 1.0]))
        assert heterostructure.match_strain(best) == pytest.approx(deformation.green_lagrange_strain.von_mises_strain, rel=1e-6)


class TestFindMatches:
    def test_sorted_by_ascending_strain(self, matches):
        strains = [heterostructure.match_strain(m) for m in matches]
        assert strains == sorted(strains)

    def test_n_keep_limits_the_number_of_matches(self, film, substrate, generator):
        everything = heterostructure.find_matches(film, substrate, generator)
        assert len(everything) > 3
        assert len(heterostructure.find_matches(film, substrate, generator, 3)) == 3
        assert len(heterostructure.find_matches(film, substrate, generator, 10 ** 6)) == len(everything)

    def test_n_keep_returns_the_best_ones(self, film, substrate, generator):
        everything = heterostructure.find_matches(film, substrate, generator)
        kept = heterostructure.find_matches(film, substrate, generator, 2)
        assert [heterostructure.match_strain(m) for m in kept] == [heterostructure.match_strain(m) for m in everything[:2]]

    def test_incommensurate_pair_with_a_small_area_has_no_match(self, substrate):
        film = heterostructure.wrap_as_slab(two_layer_hex_slab(3.0 * 1.07), (1, 1, 1))
        generator = heterostructure.make_zsl_generator(30, 0.09, 0.03, 0.03)
        assert heterostructure.find_matches(film, substrate, generator) == []

    def test_matches_respect_the_maximum_area(self, film, substrate):
        generator = heterostructure.make_zsl_generator(60, 0.09, 0.03, 0.03)
        for match in heterostructure.find_matches(film, substrate, generator):
            assert match.match_area <= 60 + 1e-6

    def test_swapping_film_and_substrate_gives_the_same_number_of_matches(self, film, substrate, generator):
        assert len(heterostructure.find_matches(film, substrate, generator)) == \
               len(heterostructure.find_matches(substrate, film, generator))


class TestExplainNoMatch:
    def test_reports_how_far_the_pair_is_from_matching(self, substrate, capsys):
        film = heterostructure.wrap_as_slab(two_layer_hex_slab(3.0 * 1.07), (1, 1, 1))
        heterostructure.explain_no_match(film, substrate, max_area=30, max_area_ratio_tol=0.09)
        out = capsys.readouterr().out
        assert out.startswith('No match with the configured settings')
        assert 'max_area=30 Å²: no match up to a 15% tolerance' in out
        assert 'max_area=60 Å², tolerance=10%:' in out and 'lowest strain' in out and 'match(es)' in out

    def test_hopeless_pair(self, substrate, capsys):
        film = heterostructure.wrap_as_slab(two_layer_hex_slab(3.0 * 1.5), (1, 1, 1))
        heterostructure.explain_no_match(film, substrate, max_area=15, max_area_ratio_tol=0.09)
        out = capsys.readouterr().out
        assert out.count('no match up to a 15% tolerance') == 2

    def test_does_not_change_the_slabs(self, substrate, capsys):
        film = heterostructure.wrap_as_slab(two_layer_hex_slab(3.0 * 1.07), (1, 1, 1))
        before = film.lattice.matrix.copy()
        heterostructure.explain_no_match(film, substrate, max_area=30, max_area_ratio_tol=0.09)
        np.testing.assert_array_equal(film.lattice.matrix, before)


class TestInPlaneFrame:
    def test_orthonormal_basis_with_u_along_a(self):
        slab = heterostructure.wrap_as_slab(rotate_about_z(oblique_slab(), 41.0), (0, 0, 1))
        frame = heterostructure._in_plane_frame(slab)
        assert frame.shape == (3, 2)
        np.testing.assert_allclose(frame.T @ frame, np.eye(2), atol=1e-12)
        a = slab.lattice.matrix[0]
        np.testing.assert_allclose(frame[:, 0], a / np.linalg.norm(a))
        normal = np.cross(*slab.lattice.matrix[:2])
        np.testing.assert_allclose(frame.T @ normal, 0.0, atol=1e-9)


class TestEffectiveTwist:
    @staticmethod
    def match_with_rotation(angle_degrees, reflect=False):
        t = np.radians(angle_degrees)
        film_vectors = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        rotation = np.array([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]])
        substrate_vectors = film_vectors @ rotation
        if reflect:
            substrate_vectors = substrate_vectors[::-1]
        return SimpleNamespace(film_sl_vectors=film_vectors, substrate_sl_vectors=substrate_vectors)

    @staticmethod
    def square_slab_pair():
        slab = heterostructure.wrap_as_slab(Structure(Lattice.tetragonal(3.0, 20.0), ['Cu'], [[0, 0, 0]]), (0, 0, 1))
        return slab, slab

    @pytest.mark.parametrize('angle', [0.0, 20.0, -35.0, 90.0])
    def test_pure_rotation_is_returned_as_the_twist(self, angle):
        film_slab, substrate_slab = self.square_slab_pair()
        twist = heterostructure.effective_twist(self.match_with_rotation(angle), film_slab, substrate_slab)
        assert twist == pytest.approx(angle)

    def test_identity_match_has_no_twist(self, substrate):
        match = SimpleNamespace(film_sl_vectors=substrate.lattice.matrix[:2], substrate_sl_vectors=substrate.lattice.matrix[:2])
        assert heterostructure.effective_twist(match, substrate, substrate) == pytest.approx(0.0, abs=1e-9)

    def test_a_reflection_has_no_defined_twist(self):
        film_slab, substrate_slab = self.square_slab_pair()
        twist = heterostructure.effective_twist(self.match_with_rotation(20.0, reflect=True), film_slab, substrate_slab)
        assert np.isnan(twist)

    def test_strain_does_not_change_the_twist(self):
        """The rotation part of (rotation x strain) is what is reported."""
        film_slab, substrate_slab = self.square_slab_pair()
        match = self.match_with_rotation(20.0)
        match.substrate_sl_vectors = match.substrate_sl_vectors * np.array([[1.04], [0.97]])     # anisotropic strain
        twist = heterostructure.effective_twist(match, film_slab, substrate_slab)
        assert twist == pytest.approx(20.0, abs=2.0)

    def test_twist_does_not_depend_on_the_cartesian_orientation_of_the_slabs(self, substrate):
        """Rotating the whole film in Cartesian space changes nothing: the twist is a property of the crystals."""
        substrate = heterostructure.wrap_as_slab(oblique_slab(), (0, 0, 1))
        generator = heterostructure.make_zsl_generator(150, 0.09, 0.05, 0.05)
        reference = None
        for angle in (0.0, 33.0, -112.0):
            film = heterostructure.wrap_as_slab(rotate_about_z(oblique_slab(scale=1.02), angle), (1, 1, 1))
            found = heterostructure.find_matches(film, substrate, generator)
            twists = sorted(round(t, 5) for t in (heterostructure.effective_twist(m, film, substrate) for m in found)
                            if np.isfinite(t))
            assert twists
            if reference is None:
                reference = twists
            assert twists == reference

    def test_matches_of_identical_hexagonal_lattices_are_multiples_of_60_degrees(self, substrate, generator):
        found = heterostructure.find_matches(substrate, substrate, generator, 20)
        twists = [heterostructure.effective_twist(m, substrate, substrate) for m in found]
        finite = [t for t in twists if np.isfinite(t)]
        assert finite
        for twist in finite:
            assert abs(((twist + 30.0) % 60.0) - 30.0) < 1e-6


# --------------------------------------------------------------------------------------------------------------
# Supercells, registries and interfaces
# --------------------------------------------------------------------------------------------------------------
class TestBuildSupercellSlabs:
    def test_in_plane_cells_are_the_ones_of_the_match(self, film, substrate, matches):
        for match in matches:
            film_sl, substrate_sl = heterostructure.build_supercell_slabs(film, substrate, match)
            np.testing.assert_allclose(film_sl.lattice.matrix[:2], match.film_sl_vectors, atol=1e-8)
            np.testing.assert_allclose(substrate_sl.lattice.matrix[:2], match.substrate_sl_vectors, atol=1e-8)

    def test_number_of_atoms_follows_the_supercell_area(self, film, substrate, matches):
        for match in matches:
            film_sl, substrate_sl = heterostructure.build_supercell_slabs(film, substrate, match)
            assert len(film_sl) == len(film) * round(area_of(film_sl) / area_of(film))
            assert len(substrate_sl) == len(substrate) * round(area_of(substrate_sl) / area_of(substrate))
            assert area_of(film_sl) == pytest.approx(match.match_area)

    def test_the_out_of_plane_vector_is_untouched(self, film, substrate, matches):
        film_sl, substrate_sl = heterostructure.build_supercell_slabs(film, substrate, matches[0])
        np.testing.assert_allclose(film_sl.lattice.matrix[2], film.lattice.matrix[2])
        np.testing.assert_allclose(substrate_sl.lattice.matrix[2], substrate.lattice.matrix[2])

    def test_atoms_keep_their_composition_and_density(self, film, substrate, matches):
        film_sl, _ = heterostructure.build_supercell_slabs(film, substrate, matches[0])
        n = len(film_sl) // len(film)
        assert film_sl.composition == film.composition * n
        assert film_sl.density == pytest.approx(film.density)

    def test_input_slabs_are_not_modified(self, film, substrate, matches):
        film_lattice, substrate_lattice = film.lattice.matrix.copy(), substrate.lattice.matrix.copy()
        heterostructure.build_supercell_slabs(film, substrate, matches[0])
        np.testing.assert_array_equal(film.lattice.matrix, film_lattice)
        np.testing.assert_array_equal(substrate.lattice.matrix, substrate_lattice)
        assert len(film) == 2 and len(substrate) == 2


class TestSubsampleEvenly:
    def test_short_lists_are_returned_whole(self):
        items = [3, 1, 2]
        result = heterostructure.subsample_evenly(items, 5)
        assert result == items and result is not items
        assert heterostructure.subsample_evenly(items, 3) == items

    @pytest.mark.parametrize('n_keep', [2, 3, 5, 7])
    def test_exactly_n_keep_ordered_items_including_both_ends(self, n_keep):
        items = list(range(20))
        result = heterostructure.subsample_evenly(items, n_keep)
        assert len(result) == n_keep
        assert result == sorted(set(result))
        assert result[0] == 0 and result[-1] == 19

    def test_spacing_is_even(self):
        result = heterostructure.subsample_evenly(list(range(101)), 6)
        assert result == [0, 20, 40, 60, 80, 100]
        assert np.ptp(np.diff(heterostructure.subsample_evenly(list(range(23)), 5))) <= 1

    def test_single_item_is_the_first(self):
        assert heterostructure.subsample_evenly(['a', 'b', 'c'], 1) == ['a']

    def test_zero_gives_an_empty_list(self):
        assert heterostructure.subsample_evenly([1, 2, 3], 0) == []

    def test_works_with_any_objects(self):
        items = [(0.0, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7)]
        assert heterostructure.subsample_evenly(items, 2) == [items[0], items[-1]]


class TestEnumerateRegistries:
    def registries(self, supercells, n_keep=100, **kwargs):
        film_sl, substrate_sl = supercells
        return heterostructure.enumerate_registries(film_sl, substrate_sl, n_keep, reference_gap=3.0,
                                                    vacuum_over_film=10.0, **kwargs)

    def test_shifts_are_fractional_and_sorted(self, supercells):
        registries = self.registries(supercells)
        assert registries and registries == sorted(registries)
        assert all(0.0 <= x < 1.0 and 0.0 <= y < 1.0 for x, y in registries)

    def test_n_keep_is_a_maximum(self, supercells):
        everything = self.registries(supercells)
        assert len(everything) > 2
        subset = self.registries(supercells, n_keep=2)
        assert len(subset) == 2
        assert set(subset) <= set(everything)
        assert subset[0] == everything[0] and subset[-1] == everything[-1]

    def test_registries_are_distinct_modulo_the_lattice(self, supercells):
        registries = np.array(self.registries(supercells))
        a, b = supercells[1].lattice.matrix[:2]
        images = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        for index, shift in enumerate(registries):
            for other in registries[index + 1:]:
                distance = min(np.linalg.norm((shift - other + np.array(image)) @ np.array([a, b])) for image in images)
                assert distance >= 0.1 - 1e-6

    def test_is_deterministic(self, supercells):
        assert self.registries(supercells) == self.registries(supercells)

    def test_translations_of_the_lattice_and_near_duplicates_are_merged(self, supercells, monkeypatch):
        candidates = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 2.0), (0.5, 0.5), (0.5, 0.5000001), (0.5, 0.0)]
        monkeypatch.setattr(Interface, 'get_shifts_based_on_adsorbate_sites', lambda self, tolerance: candidates)
        assert self.registries(supercells) == [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]

    def test_falls_back_to_the_origin_when_no_site_is_found(self, supercells, monkeypatch):
        monkeypatch.setattr(Interface, 'get_shifts_based_on_adsorbate_sites', lambda self, tolerance: [])
        assert self.registries(supercells) == [(0.0, 0.0)]

    def test_larger_tolerance_merges_more_registries(self, supercells, monkeypatch):
        candidates = [(0.0, 0.0), (0.02, 0.0), (0.2, 0.0), (0.21, 0.0), (0.5, 0.5)]
        monkeypatch.setattr(Interface, 'get_shifts_based_on_adsorbate_sites', lambda self, tolerance: candidates)
        fine = self.registries(supercells, tolerance=0.01)
        coarse = self.registries(supercells, tolerance=0.5)
        assert len(coarse) < len(fine)


class TestBuildInterface:
    @staticmethod
    def labels(interface):
        return np.array(interface.site_properties['interface_label'])

    def test_contains_all_atoms_with_their_labels(self, supercells):
        film_sl, substrate_sl = supercells
        interface = heterostructure.build_interface(film_sl, substrate_sl, (0.0, 0.0), 3.0, 10.0)
        assert isinstance(interface, Interface)
        assert (self.labels(interface) == 'film').sum() == len(film_sl)
        assert (self.labels(interface) == 'substrate').sum() == len(substrate_sl)

    def test_in_plane_cell_is_the_substrate_cell_up_to_a_rigid_rotation(self, supercells):
        """pymatgen re-orients the interface cell (a along x): lengths, angle and area are those of the substrate."""
        film_sl, substrate_sl = supercells
        interface = heterostructure.build_interface(film_sl, substrate_sl, (0.0, 0.0), 3.0, 10.0)
        interface_cell, substrate_cell = interface.lattice.matrix[:2], substrate_sl.lattice.matrix[:2]
        np.testing.assert_allclose(interface_cell @ interface_cell.T, substrate_cell @ substrate_cell.T, atol=1e-8)
        assert area_of(interface) == pytest.approx(area_of(substrate_sl))

    def test_gap_between_substrate_and_film(self, supercells):
        film_sl, substrate_sl = supercells
        for gap in (2.0, 3.5):
            interface = heterostructure.build_interface(film_sl, substrate_sl, (0.0, 0.0), gap, 10.0)
            z, labels = interface.cart_coords[:, 2], self.labels(interface)
            assert z[labels == 'film'].min() - z[labels == 'substrate'].max() == pytest.approx(gap, abs=1e-6)

    def test_vacuum_over_the_film(self, supercells):
        film_sl, substrate_sl = supercells
        small = heterostructure.build_interface(film_sl, substrate_sl, (0.0, 0.0), 3.0, 10.0)
        large = heterostructure.build_interface(film_sl, substrate_sl, (0.0, 0.0), 3.0, 20.0)
        assert large.lattice.c - small.lattice.c == pytest.approx(10.0, abs=1e-6)

    def test_registry_shifts_the_film_and_nothing_else(self, supercells):
        """Regression: Interface.from_slabs ignores in_plane_offset, so the shift is applied explicitly."""
        film_sl, substrate_sl = supercells
        reference = heterostructure.build_interface(film_sl, substrate_sl, (0.0, 0.0), 3.0, 10.0)
        shifted = heterostructure.build_interface(film_sl, substrate_sl, (0.3, 0.1), 3.0, 10.0)
        film = self.labels(reference) == 'film'
        assert not np.allclose(reference.frac_coords, shifted.frac_coords)
        np.testing.assert_allclose(shifted.frac_coords[~film], reference.frac_coords[~film], atol=1e-9)
        difference = periodic_difference(shifted.frac_coords[film], reference.frac_coords[film])
        np.testing.assert_allclose(difference, np.tile([0.3, 0.1, 0.0], (film.sum(), 1)), atol=1e-9)

    def test_a_full_lattice_translation_is_the_same_registry(self, supercells):
        film_sl, substrate_sl = supercells
        reference = heterostructure.build_interface(film_sl, substrate_sl, (0.0, 0.0), 3.0, 10.0)
        translated = heterostructure.build_interface(film_sl, substrate_sl, (1.0, -1.0), 3.0, 10.0)
        np.testing.assert_allclose(periodic_difference(translated.frac_coords, reference.frac_coords), 0.0, atol=1e-9)

    def test_film_atoms_are_wrapped_into_the_cell(self, supercells):
        film_sl, substrate_sl = supercells
        interface = heterostructure.build_interface(film_sl, substrate_sl, (0.9, 0.8), 3.0, 10.0)
        assert np.all(interface.frac_coords >= -1e-9) and np.all(interface.frac_coords < 1 + 1e-9)

    def test_input_supercells_are_not_modified(self, supercells):
        film_sl, substrate_sl = supercells
        film_before, substrate_before = film_sl.frac_coords.copy(), substrate_sl.frac_coords.copy()
        heterostructure.build_interface(film_sl, substrate_sl, (0.3, 0.1), 3.0, 10.0)
        np.testing.assert_array_equal(film_sl.frac_coords, film_before)
        np.testing.assert_array_equal(substrate_sl.frac_coords, substrate_before)


class TestSplitInterface:
    @pytest.fixture
    def interface(self, supercells):
        return heterostructure.build_interface(*supercells, (0.2, 0.4), 3.0, 10.0)

    def test_film_and_substrate_atoms_in_the_interface_cell(self, interface):
        film_part, substrate_part = heterostructure.split_interface(interface)
        labels = np.array(interface.site_properties['interface_label'])
        assert len(film_part) == (labels == 'film').sum()
        assert len(substrate_part) == (labels == 'substrate').sum()
        assert film_part.lattice == interface.lattice and substrate_part.lattice == interface.lattice

    def test_positions_and_species_are_those_of_the_interface(self, interface):
        film_part, substrate_part = heterostructure.split_interface(interface)
        labels = np.array(interface.site_properties['interface_label'])
        np.testing.assert_allclose(film_part.frac_coords, interface.frac_coords[labels == 'film'])
        np.testing.assert_allclose(substrate_part.frac_coords, interface.frac_coords[labels == 'substrate'])
        assert [s.symbol for s in film_part.species] == [s.symbol for s, l in zip(interface.species, labels) if l == 'film']

    def test_parts_add_up_to_the_interface(self, interface):
        film_part, substrate_part = heterostructure.split_interface(interface)
        assert film_part.composition + substrate_part.composition == interface.composition

    def test_parts_are_plain_structures(self, interface):
        for part in heterostructure.split_interface(interface):
            assert type(part) is Structure and not part.site_properties


# --------------------------------------------------------------------------------------------------------------
# Writing structures
# --------------------------------------------------------------------------------------------------------------
class TestPlainAndRightHandedStructures:
    def test_to_plain_structure_strips_the_site_properties(self, supercells):
        interface = heterostructure.build_interface(*supercells, (0.0, 0.0), 3.0, 10.0)
        plain = heterostructure.to_plain_structure(interface)
        assert type(plain) is Structure and not plain.site_properties
        np.testing.assert_allclose(plain.frac_coords, interface.frac_coords)
        assert plain.lattice == interface.lattice

    @staticmethod
    def left_handed():
        lattice = Lattice([[0, 3, 0], [3, 0, 0], [0, 0, 20]])
        assert np.linalg.det(lattice.matrix) < 0
        return Structure(lattice, ['Cu', 'O'], [[0.1, 0.2, 0.1], [0.4, 0.7, 0.2]])

    def test_left_handed_cell_becomes_right_handed_without_moving_the_atoms(self):
        structure = self.left_handed()
        fixed = heterostructure.to_right_handed(structure)
        assert np.linalg.det(fixed.lattice.matrix) > 0
        np.testing.assert_allclose(fixed.cart_coords, structure.cart_coords, atol=1e-9)
        assert [s.symbol for s in fixed.species] == [s.symbol for s in structure.species]
        assert area_of(fixed) == pytest.approx(area_of(structure))
        assert fixed.volume == pytest.approx(structure.volume)

    def test_right_handed_cell_is_returned_unchanged(self):
        structure = two_layer_hex_slab()
        fixed = heterostructure.to_right_handed(structure)
        assert fixed.lattice == structure.lattice
        np.testing.assert_allclose(fixed.frac_coords, structure.frac_coords)

    def test_supercell_slabs_of_a_match_can_be_left_handed_and_are_fixed(self, film, substrate, matches):
        handedness = []
        for match in matches:
            for slab in heterostructure.build_supercell_slabs(film, substrate, match):
                handedness.append(np.linalg.det(slab.lattice.matrix) > 0)
                assert np.linalg.det(heterostructure.to_right_handed(slab).lattice.matrix) > 0
        assert not all(handedness)          # the situation this function exists for does occur with ZSL matches


class TestWriteStructure:
    def test_writes_a_readable_poscar_creating_the_folders(self, tmp_path):
        path = str(tmp_path / 'match_0' / 'registry_0' / 'POSCAR')
        heterostructure.write_structure(two_layer_hex_slab(), path)
        atoms = read_vasp(path)
        assert atoms.get_chemical_formula() == 'OTi'
        np.testing.assert_allclose(atoms.cell.array, two_layer_hex_slab().lattice.matrix, atol=1e-8)

    def test_path_without_a_directory(self, tmp_path, monkeypatch):
        """Regression: os.makedirs('') raised FileNotFoundError for a bare file name."""
        monkeypatch.chdir(tmp_path)
        heterostructure.write_structure(two_layer_hex_slab(), 'POSCAR')
        assert (tmp_path / 'POSCAR').exists()

    def test_existing_folder_is_fine(self, tmp_path):
        heterostructure.write_structure(two_layer_hex_slab(), str(tmp_path / 'POSCAR'))
        heterostructure.write_structure(two_layer_hex_slab(), str(tmp_path / 'POSCAR'))

    def test_interfaces_with_site_properties_can_be_written(self, supercells, tmp_path):
        interface = heterostructure.build_interface(*supercells, (0.0, 0.0), 3.0, 10.0)
        heterostructure.write_structure(interface, str(tmp_path / 'POSCAR'))
        assert len(read_vasp(str(tmp_path / 'POSCAR'))) == len(interface)

    def test_written_cell_is_right_handed_and_positions_are_preserved(self, tmp_path):
        structure = TestPlainAndRightHandedStructures.left_handed()
        heterostructure.write_structure(structure, str(tmp_path / 'POSCAR'))
        atoms = read_vasp(str(tmp_path / 'POSCAR'))
        assert np.linalg.det(atoms.cell.array) > 0
        written = Structure.from_file(str(tmp_path / 'POSCAR'))
        for site in structure:
            assert min(np.linalg.norm(written.cart_coords - site.coords, axis=1)) < 1e-6

    def test_atoms_are_sorted_by_species(self, tmp_path):
        structure = Structure(Lattice.cubic(10.0), ['O', 'Ti', 'O', 'Ti'],
                              [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]])
        heterostructure.write_structure(structure, str(tmp_path / 'POSCAR'))
        assert read_vasp(str(tmp_path / 'POSCAR')).get_chemical_symbols() == ['O', 'O', 'Ti', 'Ti']


# --------------------------------------------------------------------------------------------------------------
# Energies
# --------------------------------------------------------------------------------------------------------------
class TestAdhesionEnergy:
    def test_formula_and_unit_conversion(self):
        # (E_interface - E_film - E_substrate) / A with 1 eV/A^2 = 16.0218 J/m^2
        assert heterostructure.get_adhesion_energy_of_formation(-20.5, -10.0, -10.0, 5.0) == pytest.approx(-0.5 / 5.0 * 16.0218)

    def test_zero_for_non_interacting_parts(self):
        assert heterostructure.get_adhesion_energy_of_formation(-20.0, -10.0, -10.0, 8.0) == 0.0

    def test_binding_is_negative_and_repulsion_positive(self):
        assert heterostructure.get_adhesion_energy_of_formation(-21.0, -10.0, -10.0, 8.0) < 0
        assert heterostructure.get_adhesion_energy_of_formation(-19.0, -10.0, -10.0, 8.0) > 0

    def test_normalised_by_one_interface_not_two_surfaces(self):
        adhesion = heterostructure.get_adhesion_energy_of_formation(-21.0, -10.0, -10.0, 8.0)
        assert adhesion == pytest.approx(-1.0 / 8.0 * 16.0218)


class TestAdhesionEnergyWithAPotential:
    """End to end with a real (EMT) potential: build, write, and evaluate interface and references."""

    @staticmethod
    def energies(tmp_path, gap, emt_backend, registry=(0.0, 0.0)):
        film = heterostructure.wrap_as_slab(two_layer_hex_slab(3.0 * 1.01, ('Cu', 'Cu')), (1, 1, 1))
        substrate = heterostructure.wrap_as_slab(two_layer_hex_slab(3.0, ('Al', 'Al')), (0, 0, 1))
        generator = heterostructure.make_zsl_generator(100, 0.09, 0.03, 0.03)
        match = heterostructure.find_matches(film, substrate, generator, 1)[0]
        film_sl, substrate_sl = heterostructure.build_supercell_slabs(film, substrate, match)
        interface = heterostructure.build_interface(film_sl, substrate_sl, registry, gap, 12.0)
        film_part, substrate_part = heterostructure.split_interface(interface)
        folder = tmp_path / f'gap_{gap}'
        values = {}
        for name, structure in [('interface', interface), ('film', film_part), ('substrate', substrate_part)]:
            heterostructure.write_structure(structure, str(folder / name / 'POSCAR'))
            values[name] = mlip.single_shot_energy_calculation(str(folder / name / 'POSCAR'))[0]
        area = np.linalg.norm(np.cross(interface.lattice.matrix[0], interface.lattice.matrix[1]))
        return heterostructure.get_adhesion_energy_of_formation(values['interface'], values['film'], values['substrate'], area)

    def test_adhesion_vanishes_when_the_parts_are_far_apart(self, tmp_path, emt_backend):
        assert self.energies(tmp_path, 14.0, emt_backend) == pytest.approx(0.0, abs=1e-6)

    def test_close_parts_interact(self, tmp_path, emt_backend):
        close = self.energies(tmp_path, 2.5, emt_backend)
        assert abs(close) > 1e-3
        assert close != pytest.approx(self.energies(tmp_path, 14.0, emt_backend), abs=1e-3)

    def test_registry_changes_the_adhesion_energy(self, tmp_path, emt_backend):
        origin = self.energies(tmp_path / 'a', 2.5, emt_backend, registry=(0.0, 0.0))
        shifted = self.energies(tmp_path / 'b', 2.5, emt_backend, registry=(0.5, 0.0))
        assert origin != pytest.approx(shifted, abs=1e-4)
