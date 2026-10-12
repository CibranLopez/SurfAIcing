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
