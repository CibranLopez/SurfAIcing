import json
import os

import numpy as np
import pytest

from pymatgen.core import Lattice, Structure

import libraries.adsorption as adsorption
from tests.helpers import fcc111_slab, square_slab, write_poscar


def frame_of(structure, surface='top'):
    return adsorption._surface_frame(structure, surface=surface)


def sites_of(structure, surface='top', **kwargs):
    unwrapped, normal, u, v = frame_of(structure, surface)
    return adsorption.voronoi_adsorption_sites(unwrapped, normal, u, v, **kwargs)


def kinds(sites):
    return sorted(site['kind'] for site in sites)


def rocksalt_001_slab():
    """NaCl(001) bilayer: Na and Cl share the top layer."""
    return Structure(Lattice.tetragonal(4.2, 20.0), ['Na', 'Cl', 'Cl', 'Na'],
                     [[0, 0, 0.2], [0.5, 0.5, 0.2], [0.5, 0.5, 0.1], [0, 0, 0.1]])


def tilted_slab():
    """Cell whose c axis is far from the surface normal (as for a general Miller index)."""
    return Structure(Lattice([[4.0, 0.0, 1.0], [0.0, 4.0, 1.0], [2.0, 2.0, 4.0]]), ['Ti'] * 3,
                     [[0.1, 0.1, 0.8], [0.5, 0.2, 0.8], [0.2, 0.7, 0.8]])


# --------------------------------------------------------------------------------------------------------------
# H-H distance
# --------------------------------------------------------------------------------------------------------------
class TestHydrogenPairDistance:
    @staticmethod
    def with_hydrogens(tmp_path, positions, extra_species=()):
        lattice = Lattice.cubic(10.0)
        species = list(extra_species) + ['H'] * len(positions)
        coords = [[0.5, 0.5, 0.5]] * len(extra_species) + list(positions)
        folder = tmp_path / 'conf'
        write_poscar(Structure(lattice, species, coords), str(folder))
        return str(folder)

    def test_molecular_distance(self, tmp_path):
        folder = self.with_hydrogens(tmp_path, [[0.1, 0.1, 0.1], [0.1, 0.1, 0.174]])
        assert adsorption.get_hydrogen_pair_distance(folder) == pytest.approx(0.74)

    def test_periodic_images_are_taken_into_account(self, tmp_path):
        folder = self.with_hydrogens(tmp_path, [[0.02, 0.5, 0.5], [0.98, 0.5, 0.5]])
        assert adsorption.get_hydrogen_pair_distance(folder) == pytest.approx(0.4)

    def test_dissociated_molecule(self, tmp_path):
        folder = self.with_hydrogens(tmp_path, [[0.1, 0.1, 0.1], [0.3, 0.1, 0.1]])
        assert adsorption.get_hydrogen_pair_distance(folder) == pytest.approx(2.0)

    def test_a_single_hydrogen_gives_none(self, tmp_path):
        assert adsorption.get_hydrogen_pair_distance(self.with_hydrogens(tmp_path, [[0.1, 0.1, 0.1]])) is None

    def test_three_hydrogens_give_none(self, tmp_path):
        folder = self.with_hydrogens(tmp_path, [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
        assert adsorption.get_hydrogen_pair_distance(folder) is None

    def test_other_species_are_ignored(self, tmp_path):
        folder = self.with_hydrogens(tmp_path, [[0.1, 0.1, 0.1], [0.1, 0.1, 0.174]], extra_species=['Cu', 'Cu'])
        assert adsorption.get_hydrogen_pair_distance(folder) == pytest.approx(0.74)

    def test_folder_without_structure_gives_none(self, tmp_path):
        assert adsorption.get_hydrogen_pair_distance(str(tmp_path)) is None


# --------------------------------------------------------------------------------------------------------------
# Surface frame
# --------------------------------------------------------------------------------------------------------------
class TestSurfaceFrame:
    def test_normal_is_perpendicular_to_the_surface_for_any_cell(self):
        structure = tilted_slab()
        _, normal, u, v = frame_of(structure)
        a, b, c = structure.lattice.matrix
        assert np.linalg.norm(normal) == pytest.approx(1.0)
        assert normal @ a == pytest.approx(0.0, abs=1e-12)
        assert normal @ b == pytest.approx(0.0, abs=1e-12)
        assert normal @ c > 0                                       # oriented towards the vacuum side of c

    def test_in_plane_basis_is_orthonormal(self):
        _, normal, u, v = frame_of(tilted_slab())
        for first, second in [(u, v), (u, normal), (v, normal)]:
            assert first @ second == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(u) == pytest.approx(1.0) and np.linalg.norm(v) == pytest.approx(1.0)

    def test_u_is_along_the_first_lattice_vector(self):
        structure = tilted_slab()
        _, _, u, _ = frame_of(structure)
        np.testing.assert_allclose(u, structure.lattice.matrix[0] / np.linalg.norm(structure.lattice.matrix[0]))

    def test_bottom_surface_flips_the_normal(self):
        structure = fcc111_slab()
        _, top_normal, _, _ = frame_of(structure, 'top')
        _, bottom_normal, _, _ = frame_of(structure, 'bottom')
        np.testing.assert_allclose(bottom_normal, -top_normal)

    def test_left_handed_cell_still_points_the_normal_towards_c(self):
        lattice = Lattice([[3, 0, 0], [0, -3, 0], [0, 0, 20]])      # det < 0: a x b points along -z
        structure = Structure(lattice, ['Cu'], [[0, 0, 0.2]])
        _, normal, _, _ = frame_of(structure)
        assert normal @ lattice.matrix[2] > 0

    def test_slab_is_unwrapped_when_it_crosses_the_cell_boundary(self):
        structure = fcc111_slab()
        structure.translate_sites(range(len(structure)), [0, 0, 0.9], frac_coords=True, to_unit_cell=True)
        original_c = structure.frac_coords[:, 2] % 1.0
        assert original_c.max() - original_c.min() > 0.5            # it is split over the boundary
        unwrapped, _, _, _ = frame_of(structure)
        assert unwrapped.frac_coords[:, 2].min() == pytest.approx(0.0, abs=1e-9)
        assert unwrapped.frac_coords[:, 2].max() < 0.5              # contiguous slab at the start of the cell

    def test_unwrapping_only_moves_the_slab_along_c(self):
        structure = fcc111_slab()
        structure.translate_sites(range(len(structure)), [0, 0, 0.9], frac_coords=True, to_unit_cell=True)
        unwrapped, _, _, _ = frame_of(structure)
        assert unwrapped.lattice == structure.lattice
        np.testing.assert_allclose(unwrapped.frac_coords[:, :2], structure.frac_coords[:, :2] % 1.0, atol=1e-9)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert unwrapped.get_distance(i, j) == pytest.approx(structure.get_distance(i, j))

    def test_invalid_surface_raises(self):
        with pytest.raises(ValueError, match="surface must be 'top' or 'bottom'"):
            frame_of(fcc111_slab(), 'side')


# --------------------------------------------------------------------------------------------------------------
# Environment fingerprints
# --------------------------------------------------------------------------------------------------------------
class TestEnvironmentSignature:
    def test_lists_species_and_distances_of_the_neighbours(self):
        structure = Structure(Lattice.cubic(20.0), ['Cu', 'O'], [[0, 0, 0], [0.1, 0, 0]])
        signature = adsorption._environment_signature(structure, np.array([0.0, 0.0, 1.0]), cutoff=3.0)
        assert signature == sorted([(29, pytest.approx(1.0)), (8, pytest.approx(np.hypot(1.0, 2.0)))])

    def test_only_atoms_within_the_cutoff_are_included(self):
        structure = Structure(Lattice.cubic(20.0), ['Cu', 'Cu'], [[0, 0, 0], [0.5, 0, 0]])
        assert len(adsorption._environment_signature(structure, np.array([0.0, 0.0, 1.0]), cutoff=3.0)) == 1

    def test_periodic_images_count(self):
        structure = Structure(Lattice.cubic(4.0), ['Cu'], [[0, 0, 0]])
        signature = adsorption._environment_signature(structure, np.array([0.0, 0.0, 2.0]), cutoff=2.5)
        assert len(signature) == 2                                  # the atom and its image 4 A above


class TestSameEnvironment:
    base = [(29, 2.0), (29, 2.5), (8, 3.0)]

    def test_identical(self):
        assert adsorption._same_environment(self.base, list(self.base), 0.05)

    def test_distances_within_tolerance(self):
        assert adsorption._same_environment(self.base, [(29, 2.04), (29, 2.46), (8, 3.0)], 0.05)

    def test_distance_outside_tolerance(self):
        assert not adsorption._same_environment(self.base, [(29, 2.2), (29, 2.5), (8, 3.0)], 0.05)

    def test_different_species(self):
        assert not adsorption._same_environment(self.base, [(29, 2.0), (29, 2.5), (16, 3.0)], 0.05)

    def test_different_number_of_neighbours(self):
        assert not adsorption._same_environment(self.base, self.base[:2], 0.05)

    def test_empty_environments_are_equal(self):
        assert adsorption._same_environment([], [], 0.05)


# --------------------------------------------------------------------------------------------------------------
# Voronoi sites
# --------------------------------------------------------------------------------------------------------------
class TestVoronoiAdsorptionSites:
    def test_square_lattice_has_atop_bridge_and_hollow(self):
        assert kinds(sites_of(square_slab())) == ['atop', 'bridge', 'hollow']

    def test_fcc_111_has_atop_bridge_and_two_distinct_hollows(self):
        """fcc and hcp hollows differ only by the second layer, which the environment cutoff must reach."""
        assert kinds(sites_of(fcc111_slab())) == ['atop', 'bridge', 'hollow', 'hollow']

    def test_a_short_environment_cutoff_cannot_tell_the_two_hollows_apart(self):
        assert kinds(sites_of(fcc111_slab(), env_cutoff=2.5)) == ['atop', 'bridge', 'hollow']

    def test_species_are_never_merged(self):
        """Na and Cl atoms of the same layer give two atop sites; one bridge, one hollow."""
        assert kinds(sites_of(rocksalt_001_slab())) == ['atop', 'atop', 'bridge', 'hollow']

    def test_atop_sites_lie_above_atoms_of_each_species(self):
        structure = rocksalt_001_slab()
        unwrapped, normal, u, v = frame_of(structure)
        atop = [s for s in adsorption.voronoi_adsorption_sites(unwrapped, normal, u, v) if s['kind'] == 'atop']
        below = sorted(unwrapped.get_sites_in_sphere(s['position'], 1.6)[0].specie.symbol for s in atop)
        assert below == ['Cl', 'Na']

    def test_every_site_is_at_the_adsorption_height_above_the_top_layer(self):
        structure = fcc111_slab()
        unwrapped, normal, _, _ = frame_of(structure)
        top = (unwrapped.cart_coords @ normal).max()
        for height in (1.0, 1.5, 2.2):
            sites = adsorption.voronoi_adsorption_sites(unwrapped, normal, *frame_of(structure)[2:], adsorption_height=height)
            for site in sites:
                assert site['position'] @ normal == pytest.approx(top + height)

    def test_atop_site_is_exactly_above_its_atom(self):
        structure = fcc111_slab()
        unwrapped, normal, u, v = frame_of(structure)
        atop = [s for s in adsorption.voronoi_adsorption_sites(unwrapped, normal, u, v) if s['kind'] == 'atop'][0]
        top_atom = unwrapped.cart_coords[np.argmax(unwrapped.cart_coords @ normal)]
        lateral = atop['position'] - (atop['position'] @ normal) * normal
        np.testing.assert_allclose(lateral, top_atom - (top_atom @ normal) * normal, atol=1e-8)

    def test_bridge_is_midway_between_two_nearest_neighbours_and_hollow_is_equidistant_from_three(self):
        structure = fcc111_slab(a=2.7)
        unwrapped, normal, u, v = frame_of(structure)
        top = unwrapped.cart_coords[np.argmax(unwrapped.cart_coords @ normal)]
        sites = adsorption.voronoi_adsorption_sites(unwrapped, normal, u, v, adsorption_height=0.0)
        bridge = [s for s in sites if s['kind'] == 'bridge'][0]['position']
        hollows = [s['position'] for s in sites if s['kind'] == 'hollow']
        lattice = unwrapped.lattice
        # lateral distance to the nearest top-layer atom (all periodic images)
        def nearest(position):
            images = [top + i * lattice.matrix[0] + j * lattice.matrix[1] for i in range(-2, 3) for j in range(-2, 3)]
            return sorted(np.linalg.norm(position - image) for image in images)
        assert nearest(bridge)[0] == pytest.approx(1.35)            # a / 2
        assert nearest(bridge)[0] == pytest.approx(nearest(bridge)[1])
        for hollow in hollows:
            distances = nearest(hollow)
            assert distances[0] == pytest.approx(2.7 / np.sqrt(3))
            assert distances[0] == pytest.approx(distances[2])      # three equidistant atoms

    def test_sites_are_inside_the_in_plane_cell(self):
        structure = fcc111_slab()
        unwrapped, normal, u, v = frame_of(structure)
        a, b = unwrapped.lattice.matrix[:2]
        cell = np.array([[a @ u, a @ v], [b @ u, b @ v]])
        for site in adsorption.voronoi_adsorption_sites(unwrapped, normal, u, v):
            fractional = np.array([site['position'] @ u, site['position'] @ v]) @ np.linalg.inv(cell)
            assert np.all(fractional > -1e-6) and np.all(fractional < 1 + 1e-6)

    def test_supercell_gives_the_same_inequivalent_sites(self):
        structure = fcc111_slab()
        supercell = structure.copy()
        supercell.make_supercell([2, 2, 1])
        assert kinds(sites_of(supercell)) == kinds(sites_of(structure))

    def test_supercell_with_a_non_diagonal_matrix(self):
        structure = fcc111_slab()
        supercell = structure.copy()
        supercell.make_supercell([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
        assert kinds(sites_of(supercell)) == kinds(sites_of(structure))

    def test_a_slab_crossing_the_cell_boundary_gives_the_same_sites(self):
        structure = fcc111_slab()
        shifted = structure.copy()
        shifted.translate_sites(range(len(shifted)), [0.3, 0.1, 0.9], frac_coords=True, to_unit_cell=True)
        assert kinds(sites_of(shifted)) == kinds(sites_of(structure))

    def test_uniform_3x3_grid_has_only_three_inequivalent_sites(self):
        offsets = [0.1, 0.1 + 1 / 3, 0.1 + 2 / 3]
        coords = [[x, y, 0.8] for y in offsets for x in offsets]
        structure = Structure(Lattice.tetragonal(9.0, 12.0), ['Ti'] * 9, coords)
        assert kinds(sites_of(structure)) == ['atop', 'bridge', 'hollow']

    def test_non_uniform_grid_keeps_the_inequivalent_atoms(self):
        """Spacings of 2.7, 2.7 and 3.6 A along each axis (the layout of the previous test-suite): the atoms are
        no longer all equivalent. In each direction an atom is either 'central' or 'at the end' (the two ends are
        mirror images), so there are (end, end), (end, centre) and (centre, centre) atoms: three atop sites."""
        offsets = [0.15, 0.45, 0.75]
        coords = [[x, y, 0.8] for y in offsets for x in offsets]
        structure = Structure(Lattice.tetragonal(9.0, 12.0), ['Ti'] * 9, coords)
        found = kinds(sites_of(structure))
        assert found.count('atop') == 3
        assert found.count('hollow') == 3 and found.count('bridge') == 4

    def test_top_layer_is_defined_by_z_tolerance(self):
        """A rumpled top layer: with a small tolerance only the highest atom defines the sites."""
        structure = Structure(Lattice.tetragonal(4.0, 20.0), ['Cu', 'Cu', 'Cu'],
                              [[0, 0, 0.30], [0.5, 0.5, 0.28], [0.5, 0.5, 0.15]])        # 0.4 A apart along z
        narrow = sites_of(structure, z_tolerance=0.1)
        wide = sites_of(structure, z_tolerance=0.6)
        assert kinds(narrow) == ['atop', 'bridge', 'hollow']
        assert kinds(wide).count('atop') == 2

    def test_bottom_termination_sites_are_below_the_slab(self):
        structure = fcc111_slab()
        unwrapped, normal, u, v = adsorption._surface_frame(structure, 'bottom')
        heights = unwrapped.cart_coords @ normal                     # along the flipped normal
        for site in adsorption.voronoi_adsorption_sites(unwrapped, normal, u, v, adsorption_height=1.5):
            assert site['position'] @ normal == pytest.approx(heights.max() + 1.5)
        _, top_normal, _, _ = adsorption._surface_frame(structure, 'top')
        for site in adsorption.voronoi_adsorption_sites(unwrapped, normal, u, v):
            assert site['position'] @ top_normal < (unwrapped.cart_coords @ top_normal).min()

    def test_atop_is_always_returned_first_so_that_it_keeps_its_label(self):
        sites = sites_of(fcc111_slab())
        assert [s['kind'] for s in sites][0] == 'atop'
        order = {'atop': 0, 'bridge': 1, 'hollow': 2}
        assert [order[s['kind']] for s in sites] == sorted(order[s['kind']] for s in sites)

    def test_sites_are_cartesian_three_vectors(self):
        for site in sites_of(fcc111_slab()):
            assert np.shape(site['position']) == (3,)


# --------------------------------------------------------------------------------------------------------------
# Generation of the configurations
# --------------------------------------------------------------------------------------------------------------
@pytest.fixture
def surface_dir(tmp_path):
    folder = tmp_path / 'surface'
    write_poscar(fcc111_slab(), str(folder))
    (folder / 'KPOINTS').write_text('Automatic mesh\n0\nGamma\n 10 10 1\n')
    (folder / 'INCAR').write_text('ENCUT = 400\n')
    (folder / 'POTCAR').write_text('fake potcar\n')
    (folder / 'run.sh').write_text('#!/bin/bash\n')
    return folder


def conf_names(paths):
    return sorted(os.path.basename(path) for path in paths)


class TestGenerateHydrogenSites:
    def test_one_folder_per_inequivalent_site(self, surface_dir, tmp_path):
        configs = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'))
        assert conf_names(configs) == ['conf_00_atop', 'conf_01_bridge', 'conf_02_hollow', 'conf_03_hollow']
        assert all(os.path.isdir(path) for path in configs)

    def test_each_configuration_is_the_slab_plus_one_hydrogen(self, surface_dir, tmp_path):
        slab = Structure.from_file(str(surface_dir / 'POSCAR'))
        for path in adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out')):
            structure = Structure.from_file(os.path.join(path, 'POSCAR'))
            assert len(structure) == len(slab) + 1
            assert structure.species[-1].symbol == 'H'
            assert structure.composition['H'] == 1
            assert structure.lattice == slab.lattice               # the cell is not modified

    def test_the_slab_itself_is_not_modified_except_for_a_rigid_shift_along_c(self, surface_dir, tmp_path):
        slab = Structure.from_file(str(surface_dir / 'POSCAR'))
        path = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'))[0]
        structure = Structure.from_file(os.path.join(path, 'POSCAR'))
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert structure.get_distance(i, j) == pytest.approx(slab.get_distance(i, j))

    def test_hydrogen_is_at_the_adsorption_height_along_the_normal(self, surface_dir, tmp_path):
        _, normal, _, _ = frame_of(Structure.from_file(str(surface_dir / 'POSCAR')))
        for height in (1.2, 1.5):
            out = tmp_path / f'out_{height}'
            for path in adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), adsorption_height=height):
                structure = Structure.from_file(os.path.join(path, 'POSCAR'))
                heights = structure.cart_coords @ normal
                assert heights[-1] - heights[:-1].max() == pytest.approx(height, abs=1e-6)

    def test_hydrogen_is_never_closer_than_min_distance_to_the_slab(self, surface_dir, tmp_path):
        for path in adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out')):
            structure = Structure.from_file(os.path.join(path, 'POSCAR'))
            distances = [structure.get_distance(len(structure) - 1, i) for i in range(len(structure) - 1)]
            assert min(distances) >= 1.0

    def test_input_files_are_copied(self, surface_dir, tmp_path):
        for path in adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out')):
            for name in ('KPOINTS', 'INCAR', 'POTCAR', 'run.sh'):
                assert open(os.path.join(path, name)).read() == open(surface_dir / name).read()

    def test_optional_input_files_may_be_missing(self, tmp_path):
        folder = tmp_path / 'bare'
        write_poscar(fcc111_slab(), str(folder))
        configs = adsorption.generate_inequivalent_hydrogen_sites(str(folder), str(tmp_path / 'out'))
        assert configs and not os.path.exists(os.path.join(configs[0], 'KPOINTS'))

    def test_sites_json_summary(self, surface_dir, tmp_path):
        out = tmp_path / 'out'
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), adsorption_height=1.7)
        summary = json.loads((out / 'sites.json').read_text())
        assert summary['surface'] == 'top' and summary['adsorbate'] == 'H' and summary['adsorption_height'] == 1.7
        assert set(summary['configurations']) == {'conf_00_atop', 'conf_01_bridge', 'conf_02_hollow', 'conf_03_hollow'}
        assert summary['configurations']['conf_01_bridge']['kind'] == 'bridge'
        assert summary['configurations']['conf_01_bridge']['orientation'] is None
        assert len(summary['configurations']['conf_01_bridge']['cartesian']) == 3

    def test_default_output_directory_is_a_sibling_called_h_absorption(self, surface_dir):
        configs = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir))
        assert os.path.dirname(configs[0]) == str(surface_dir.parent / 'H-absorption')

    def test_relative_paths_are_accepted(self, surface_dir, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        configs = adsorption.generate_inequivalent_hydrogen_sites('surface', 'relative_out')
        assert all(os.path.isabs(path) for path in configs)

    def test_custom_label(self, surface_dir, tmp_path):
        configs = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'), sample_label='site')
        assert all(os.path.basename(path).startswith('site_') for path in configs)

    def test_relaxed_contcar_is_preferred_to_poscar(self, surface_dir, tmp_path):
        write_poscar(fcc111_slab(a=2.8), str(surface_dir), 'CONTCAR')
        path = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'))[0]
        assert Structure.from_file(os.path.join(path, 'POSCAR')).lattice.a == pytest.approx(2.8)

    def test_missing_structure_raises(self, tmp_path):
        (tmp_path / 'empty').mkdir()
        with pytest.raises(FileNotFoundError, match='Neither CONTCAR nor POSCAR'):
            adsorption.generate_inequivalent_hydrogen_sites(str(tmp_path / 'empty'), str(tmp_path / 'out'))

    def test_invalid_adsorbate_raises(self, surface_dir, tmp_path):
        with pytest.raises(ValueError, match="adsorbate must be 'H' or 'H2'"):
            adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'), adsorbate='O')

    def test_invalid_surface_raises(self, surface_dir, tmp_path):
        with pytest.raises(ValueError, match="surface must be 'top' or 'bottom'"):
            adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'), surface='left')

    def test_invalid_surface_does_not_delete_existing_configurations(self, surface_dir, tmp_path):
        """Regression: with overwrite=True the old folders were removed before the argument was validated."""
        out = tmp_path / 'out'
        configs = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out))
        with pytest.raises(ValueError):
            adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), surface='left', overwrite=True)
        assert all(os.path.isdir(path) for path in configs)

    def test_invalid_surface_does_not_create_the_output_directory(self, surface_dir, tmp_path):
        with pytest.raises(ValueError):
            adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'never'), surface='left')
        assert not (tmp_path / 'never').exists()

    def test_surface_without_any_identifiable_site_raises(self, surface_dir, tmp_path, monkeypatch):
        monkeypatch.setattr(adsorption, 'voronoi_adsorption_sites', lambda *args, **kwargs: [])
        with pytest.raises(ValueError, match='Could not identify any adsorption site'):
            adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'))

    def test_hydrogen_on_the_bottom_termination_is_below_the_slab(self, surface_dir, tmp_path):
        _, normal, _, _ = frame_of(Structure.from_file(str(surface_dir / 'POSCAR')))
        path = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'), surface='bottom')[0]
        structure = Structure.from_file(os.path.join(path, 'POSCAR'))
        heights = structure.cart_coords @ normal
        assert heights[-1] == pytest.approx(heights[:-1].min() - 1.5, abs=1e-6)
        summary = json.loads((tmp_path / 'out' / 'sites.json').read_text())
        assert summary['surface'] == 'bottom'

    def test_top_and_bottom_of_an_asymmetric_slab_differ(self, tmp_path):
        slab = Structure(Lattice.tetragonal(3.0, 20.0), ['Cu', 'Au'], [[0, 0, 0.1], [0, 0, 0.2]])
        folder = tmp_path / 'asym'
        write_poscar(slab, str(folder))
        top = adsorption.generate_inequivalent_hydrogen_sites(str(folder), str(tmp_path / 'top'), surface='top')
        bottom = adsorption.generate_inequivalent_hydrogen_sites(str(folder), str(tmp_path / 'bottom'), surface='bottom')
        def nearest_species(path):
            structure = Structure.from_file(os.path.join(path, 'POSCAR'))
            hydrogen = len(structure) - 1
            return min(range(hydrogen), key=lambda i: structure.get_distance(i, hydrogen))
        top_neighbour = Structure.from_file(os.path.join(top[0], 'POSCAR'))[nearest_species(top[0])].specie.symbol
        bottom_neighbour = Structure.from_file(os.path.join(bottom[0], 'POSCAR'))[nearest_species(bottom[0])].specie.symbol
        assert (top_neighbour, bottom_neighbour) == ('Au', 'Cu')

    def test_oblique_cell_places_hydrogen_along_the_surface_normal_not_along_z(self, tmp_path):
        folder = tmp_path / 'tilted'
        slab = tilted_slab()
        write_poscar(slab, str(folder))
        configs = adsorption.generate_inequivalent_hydrogen_sites(str(folder), str(tmp_path / 'out'), z_tolerance=0.5)
        assert configs
        _, normal, _, _ = frame_of(slab)
        for path in configs:
            structure = Structure.from_file(os.path.join(path, 'POSCAR'))
            heights = structure.cart_coords @ normal
            assert heights[-1] - heights[:-1].max() == pytest.approx(1.5, abs=1e-6)
        h_z = Structure.from_file(os.path.join(configs[0], 'POSCAR')).cart_coords[-1][2]
        assert abs(h_z - (max(site.z for site in slab) + 1.5)) > 0.1    # the naive "1.5 A above in z" is wrong here

    def test_slab_already_containing_hydrogen(self, tmp_path):
        slab = fcc111_slab()
        slab.append('H', [0.5, 0.5, 0.0])
        folder = tmp_path / 'with_h'
        write_poscar(slab, str(folder))
        configs = adsorption.generate_inequivalent_hydrogen_sites(str(folder), str(tmp_path / 'out'))
        assert configs
        assert Structure.from_file(os.path.join(configs[0], 'POSCAR')).composition['H'] == 2


class TestOverlapFilter:
    def test_configurations_too_close_to_the_slab_are_skipped(self, surface_dir, tmp_path):
        configs = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'), min_distance=1.6)
        # an H 1.5 A above an atop atom is closer than 1.6 A, the other sites (further from every atom) are kept
        assert 'conf_00_atop' not in conf_names(configs)
        assert configs

    def test_everything_skipped_returns_an_empty_list(self, surface_dir, tmp_path):
        out = tmp_path / 'out'
        assert adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), min_distance=5.0) == []
        assert not [name for name in os.listdir(out) if name.startswith('conf')]
        assert json.loads((out / 'sites.json').read_text())['configurations'] == {}


class TestMolecularHydrogen:
    @pytest.fixture
    def configs(self, surface_dir, tmp_path):
        return adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'out'), adsorbate='H2')

    def test_both_orientations_of_each_site(self, configs):
        assert conf_names(configs) == sorted(
            f'conf_{i:02d}_{kind}_{orientation}'
            for i, kind in enumerate(['atop', 'bridge', 'hollow', 'hollow']) for orientation in ('perp', 'para'))

    def test_two_hydrogen_atoms_at_the_bond_length(self, configs):
        for path in configs:
            assert Structure.from_file(os.path.join(path, 'POSCAR')).composition['H'] == 2
            assert adsorption.get_hydrogen_pair_distance(path) == pytest.approx(0.74)

    def test_bond_length_is_configurable(self, surface_dir, tmp_path):
        configs = adsorption.generate_inequivalent_hydrogen_sites(
            str(surface_dir), str(tmp_path / 'out'), adsorbate='H2', h2_bond_length=0.9)
        assert all(adsorption.get_hydrogen_pair_distance(path) == pytest.approx(0.9) for path in configs)

    def test_orientation_relative_to_the_surface(self, configs, surface_dir):
        _, normal, _, _ = frame_of(Structure.from_file(str(surface_dir / 'POSCAR')))
        for path in configs:
            structure = Structure.from_file(os.path.join(path, 'POSCAR'))
            bond = structure.cart_coords[-1] - structure.cart_coords[-2]
            bond = bond - np.round(bond @ np.linalg.inv(structure.lattice.matrix)) @ structure.lattice.matrix
            along_normal = abs(bond @ normal) / np.linalg.norm(bond)
            if path.endswith('_perp'):
                assert along_normal == pytest.approx(1.0)
            else:
                assert along_normal == pytest.approx(0.0, abs=1e-8)

    def test_lowest_hydrogen_of_a_perpendicular_molecule_is_at_the_adsorption_height(self, configs, surface_dir):
        _, normal, _, _ = frame_of(Structure.from_file(str(surface_dir / 'POSCAR')))
        for path in [p for p in configs if p.endswith('_perp')]:
            structure = Structure.from_file(os.path.join(path, 'POSCAR'))
            heights = structure.cart_coords @ normal
            assert heights[-2] - heights[:-2].max() == pytest.approx(1.5, abs=1e-6)

    def test_summary_records_the_orientation(self, configs, tmp_path):
        summary = json.loads((tmp_path / 'out' / 'sites.json').read_text())
        assert summary['adsorbate'] == 'H2'
        assert summary['configurations']['conf_00_atop_perp']['orientation'] == 'perp'
        assert summary['configurations']['conf_00_atop_para']['orientation'] == 'para'

    def test_parallel_molecule_is_centred_on_the_site(self, surface_dir, tmp_path):
        out_h = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'h'), adsorbate='H')
        out_h2 = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(tmp_path / 'h2'), adsorbate='H2')
        single = Structure.from_file(os.path.join(out_h[0], 'POSCAR')).cart_coords[-1]
        para = Structure.from_file(os.path.join(tmp_path / 'h2' / 'conf_00_atop_para', 'POSCAR')).cart_coords[-2:]
        np.testing.assert_allclose(para.mean(axis=0), single, atol=1e-8)


class TestReuseOfExistingConfigurations:
    def test_existing_configurations_are_reused(self, surface_dir, tmp_path, capsys):
        out = tmp_path / 'out'
        first = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out))
        (out / 'conf_00_atop' / 'marker').write_text('keep me')
        second = adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out))
        assert second == first == sorted(first)
        assert (out / 'conf_00_atop' / 'marker').exists()
        assert 'already exist' in capsys.readouterr().out

    def test_overwrite_regenerates_them(self, surface_dir, tmp_path):
        out = tmp_path / 'out'
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out))
        (out / 'conf_00_atop' / 'marker').write_text('delete me')
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), overwrite=True)
        assert not (out / 'conf_00_atop' / 'marker').exists()

    def test_overwrite_removes_configurations_that_are_no_longer_generated(self, surface_dir, tmp_path):
        out = tmp_path / 'out'
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), sample_label='conf')
        (out / 'conf_99_stale').mkdir()
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), overwrite=True)
        assert not (out / 'conf_99_stale').exists()

    def test_other_folders_are_left_alone(self, surface_dir, tmp_path):
        out = tmp_path / 'out'
        (out / 'notes').mkdir(parents=True)
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), overwrite=True)
        assert (out / 'notes').exists()

    def test_reusing_with_other_settings_warns(self, surface_dir, tmp_path, capsys):
        """Regression: configurations made for another adsorbate/termination were silently reused."""
        out = tmp_path / 'out'
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), adsorbate='H', surface='top')
        capsys.readouterr()
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out), adsorbate='H2', surface='bottom',
                                                        adsorption_height=2.0)
        output = capsys.readouterr().out
        assert "adsorbate='H'" in output and "surface='top'" in output and "adsorption_height=1.5" in output

    def test_reusing_with_the_same_settings_does_not_warn(self, surface_dir, tmp_path, capsys):
        out = tmp_path / 'out'
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out))
        capsys.readouterr()
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out))
        assert '[warning]' not in capsys.readouterr().out

    def test_reusing_without_a_summary_file_does_not_fail(self, surface_dir, tmp_path):
        out = tmp_path / 'out'
        adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out))
        os.remove(out / 'sites.json')
        assert adsorption.generate_inequivalent_hydrogen_sites(str(surface_dir), str(out))
