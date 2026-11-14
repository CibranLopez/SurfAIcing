import numpy             as np
import matplotlib.pyplot as plt


def format_slab_label(
    label
):
    """Mathtext tick label of a slab folder name.

    Folder names look like ``h_k_l_i_N`` (as written by generate-slabs) or ``h_k_l_N``; they are shown as
    ``(hkl)_NN`` with the slab index zero-padded to two digits. A name in any other format is returned
    unchanged (as plain text) instead of making the whole plot fail.
    """
    parts = label.split('_')
    if len(parts) == 5 and parts[3] == 'i' and parts[4].isdigit():
        h, k, l, _, i = parts
        i = i.zfill(2)
    elif len(parts) == 4 and parts[3].isdigit():
        h, k, l, i = parts
    else:
        return label
    return f'$\\mathregular{{({h}{k}{l})_{{{i}}}}}$'


def plot_ranking(
    surface_energies_of_formation,
    filename='ranking.eps',
    figsize=(15, 5),
    ylabel=r'$E_{\text{surface}}$ (J/m$^2$)',
    dpi=50
):
    """Plot the ranking of the slabs based on their energy differences.

    Args:
        surface_energies_of_formation (pd.DataFrame): Energy differences of the slabs.
        filename      (str):          Filename to save the plot.
        figsize       (tuple):        Size of the figure.
        ylabel        (str):          Label for the y-axis.

    Returns:
        None
    """
    # Slabs whose energy could not be computed (NaN) cannot be ranked
    n_total = surface_energies_of_formation.shape[1]
    surface_energies_of_formation = surface_energies_of_formation.dropna(axis=1)
    if surface_energies_of_formation.shape[1] < n_total:
        print(f'{n_total - surface_energies_of_formation.shape[1]} slab(s) without a valid energy were left out of the plot.')
    
    if surface_energies_of_formation.shape[1] == 0:
        raise ValueError('None of the slabs has a valid (non-NaN) energy: there is nothing to plot.')

    # Sorted in ascendent order
    min_arg      = np.argsort(surface_energies_of_formation.values)[0]
    local_minima = surface_energies_of_formation.columns[min_arg]
    energies     = surface_energies_of_formation.values[0][min_arg]

    hkl = [format_slab_label(label) for label in local_minima]

    plt.figure(figsize=figsize)
    plt.plot(energies, 'o-')
    plt.xlim(-0.5, len(hkl)-1+0.5)
    plt.xticks(range(len(hkl)), hkl, rotation='vertical')
    plt.ylabel(ylabel)
    plt.ylim(bottom=min(0, 1.1*np.min(energies)))
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_ranking_comparison(
    ranking,
    filename='ranking_comparison.png',
    figsize=(15, 5),
    ylabel=r'$\gamma$ (J/m$^2$)',
    dpi=50
):
    """Plot the surface energy of every slab at 0 K and at the operating temperature, sorted by the latter.

    Args:
        ranking (pd.DataFrame): Output of ``thermal.surface_free_energy_ranking`` (index: slab folders; columns
            'surface_energy_0K_J_m2' and 'surface_energy_T_J_m2'; ``attrs`` hold the temperature and Kendall's tau).
        filename (str):         Filename to save the plot.
        figsize  (tuple):       Size of the figure.
        ylabel   (str):         Label for the y-axis.

    Returns:
        None
    """
    valid = ranking.dropna(subset=['surface_energy_T_J_m2', 'rank_T'])
    if len(valid) == 0:
        raise ValueError('None of the slabs has a finite-temperature surface energy: there is nothing to plot.')
    valid = valid.sort_values('surface_energy_T_J_m2')
    temperature = ranking.attrs.get('operating_T')
    tau = ranking.attrs.get('kendall_tau')

    hkl = [format_slab_label(label) for label in valid.index]
    x = np.arange(len(hkl))

    plt.figure(figsize=figsize)
    plt.plot(x, valid['surface_energy_T_J_m2'].values, 'o-', label=f'{temperature:g} K' if temperature is not None else 'T')
    plt.plot(x, valid['surface_energy_0K_J_m2'].values, 's--', mfc='none', label='0 K (same slabs, same order)')
    plt.xlim(-0.5, len(hkl) - 1 + 0.5)
    plt.xticks(x, hkl, rotation='vertical')
    plt.ylabel(ylabel)
    lowest = min(valid['surface_energy_T_J_m2'].min(), valid['surface_energy_0K_J_m2'].min())
    plt.ylim(bottom=min(0, 1.1 * lowest))
    if tau is not None and np.isfinite(tau):
        plt.title(f"Kendall's tau between the 0 K and the {temperature:g} K ranking: {tau:.2f}")
    plt.legend()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()

