from dataclasses import dataclass

import numpy as np


def get_regions(
        planar_potential,
        structure,
        discard_ratio,
        mode='auto'
):
    """Locate the bulk-like and vacuum regions of the planar-averaged potential of a slab along the surface normal (c axis).

    Modes:
        'midpoint': legacy. The cell is cut in two halves; the slab is ASSUMED to sit in the first half and the vacuum in
            the second (both regions are swapped if it does not).
        'auto': located from the atomic positions (slab = atoms, vacuum = largest gap along c, periodic images included).
            Valid wherever the slab sits in the cell, also when it crosses the cell boundary.

    A fraction `discard_ratio` of each region is discarded at both ends to avoid edge artifacts.

    Args:
        planar_potential (numpy.ndarray): Planar average of the electrostatic potential along c.
        structure (Structure): Slab structure (its atoms define the regions in 'auto' mode).
        discard_ratio (float): Fraction of each region discarded at its edges.
        mode (str): 'auto' or 'midpoint'.

    Returns:
        tuple: the (possibly shifted) planar potential, the (start, end) indices of the bulk-like region and the
            (start, end) indices of the vacuum region in that potential.
    """
    n = len(planar_potential)
    if mode == 'midpoint':
        mid = int(0.5 * n)
        idx_bulk = int(discard_ratio * mid)
        idx_vac  = int(discard_ratio * (n - mid))
        return planar_potential, (idx_bulk, mid - idx_bulk), (mid + idx_vac, n - idx_vac)

    if mode == 'auto':
        fractional_c = np.sort(structure.frac_coords[:, 2] % 1.0)
        gaps = np.diff(np.r_[fractional_c, fractional_c[0] + 1.0])
        k = int(np.argmax(gaps))                                    # the largest gap is the vacuum
        start = fractional_c[(k + 1) % len(fractional_c)]
        thickness = (fractional_c[k] - start) % 1.0
        planar_potential = np.roll(planar_potential, -int(round(start * n)))   # slab now starts at index 0
        t = int(round(thickness * n))
        idx_bulk = int(round(discard_ratio * t))
        idx_vac  = int(round(discard_ratio * (n - t)))
        return planar_potential, (idx_bulk, t - idx_bulk), (t + idx_vac, n - idx_vac)

    raise ValueError(f"mode must be 'midpoint' or 'auto', got {mode!r}")


@dataclass(frozen=True, eq=False)
class BandAlignment:
    """Band edges of a slab referred to the vacuum level, and the quantities they were obtained from.

    All energies are in eV (the unit of the VASP LOCPOT and of the band-structure energies).

    Attributes:
        valence_band (float): Valence band maximum with respect to the vacuum level (negative for a bound state).
        conduction_band (float): Conduction band minimum with respect to the vacuum level.
        bulk_like_potential (float): Mean planar-averaged potential in the bulk-like region of the slab.
        vacuum_potential (float): Mean planar-averaged potential in the vacuum region.
        potential (numpy.ndarray): Planar-averaged potential, rolled so that the slab starts at index 0 in 'auto' mode.
        bulk_region (tuple): (start, end) indices of the bulk-like region in `potential`.
        vacuum_region (tuple): (start, end) indices of the vacuum region in `potential`.
    """
    valence_band:        float
    conduction_band:     float
    bulk_like_potential: float
    vacuum_potential:    float
    potential:           np.ndarray
    bulk_region:         tuple
    vacuum_region:       tuple

    @property
    def ionization_potential(self):
        """IP = -VBM_vac (eV)."""
        return -self.valence_band

    @property
    def electron_affinity(self):
        """EA = -CBM_vac (eV)."""
        return -self.conduction_band


def get_band_alignment(
        planar_potential,
        structure,
        vbm_dft,
        band_gap,
        bulk_average_potential,
        discard_ratio=0.25,
        mode='auto'
):
    r"""Align the band edges of a bulk material to the vacuum level using a slab calculation:

    \begin{equation}
        VBM_{vac} = VBM_{DFT} + \left(\bar{V}_{bulk\text{-}like} - \bar{V}_{bulk}\right) - \bar{V}_{vac}, \qquad
        CBM_{vac} = VBM_{vac} + E_{gap}
    \end{equation}

    where `vbm_dft` and `band_gap` come from the bulk band structure, `bulk_average_potential` is the average
    potential of that same bulk calculation, and the slab potential is averaged over its bulk-like region and over
    its vacuum region (see ``get_regions``). The shift ``V_bulk-like - V_bulk`` transfers the bulk band edge to the
    slab's potential reference; subtracting ``V_vac`` refers it to the vacuum level.
    The ionization potential and electron affinity are ``-valence_band`` and ``-conduction_band``.

    Args:
        planar_potential (numpy.ndarray): Planar average of the slab electrostatic potential along the surface normal
            (c axis), e.g. ``Locpot.get_average_along_axis(2)``, in eV.
        structure (Structure): Slab structure (its atoms locate the regions in 'auto' mode).
        vbm_dft (float): Valence band maximum of the bulk calculation, in eV.
        band_gap (float): Band gap of the bulk calculation, in eV.
        bulk_average_potential (float): Average electrostatic potential of the bulk calculation, in eV.
        discard_ratio (float): Fraction of each region discarded at both ends to avoid edge artifacts.
        mode (str): 'auto' or 'midpoint', see ``get_regions``.

    Returns:
        BandAlignment: the band edges with respect to the vacuum level and the averaging details.

    Raises:
        ValueError: if the bulk-like or the vacuum region is empty (slab or vacuum too thin for `discard_ratio`).
    """
    potential, (b0, b1), (v0, v1) = get_regions(planar_potential, structure, discard_ratio, mode)
    if b1 <= b0:
        raise ValueError(f'The bulk-like region is empty (slab too thin for discard_ratio={discard_ratio}).')
    if v1 <= v0:
        raise ValueError(f'The vacuum region is empty (vacuum too thin for discard_ratio={discard_ratio}).')

    bulk_like = float(np.mean(potential[b0:b1]))
    vacuum    = float(np.mean(potential[v0:v1]))

    valence_band    = float(vbm_dft) + (bulk_like - float(bulk_average_potential)) - vacuum
    conduction_band = valence_band + float(band_gap)

    return BandAlignment(valence_band=valence_band, conduction_band=conduction_band,
                         bulk_like_potential=bulk_like, vacuum_potential=vacuum,
                         potential=potential, bulk_region=(b0, b1), vacuum_region=(v0, v1))
