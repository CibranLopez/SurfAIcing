import numpy             as np
import matplotlib.pyplot as plt


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
    
    # Sorted in ascendent order
    min_arg      = np.argsort(surface_energies_of_formation.values)[0]
    local_minima = surface_energies_of_formation.columns[min_arg]
    energies     = surface_energies_of_formation.values[0][min_arg]

    hkl = []
    for label in local_minima:
        sp = label.split('_')
        if len(sp) == 5:
            h, k, l, _, i = sp
            if int(i) < 10:
                i = '0' + i
        else:
            h, k, l, i = sp
        hkl.append(f'$\\mathregular{{({h}{k}{l})_{{{i}}}}}$')

    plt.figure(figsize=figsize)
    plt.plot(energies, 'o-')
    plt.xlim(-0.5, len(hkl)-1+0.5)
    plt.xticks(range(len(hkl)), hkl, rotation='vertical')
    plt.ylabel(ylabel)
    plt.ylim(bottom=min(0, 1.1*np.min(energies)))
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()