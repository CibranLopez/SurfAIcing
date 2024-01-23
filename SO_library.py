import numpy             as np
import matplotlib.pyplot as plt
import multiprocessing   as mp
import re
import os
import subprocess

from sklearn.model_selection import learning_curve
from sklearn.preprocessing   import StandardScaler

import sys
sys.path.append('../../ICMAB/')
import Diffusion.D_library as DL

linewidth    = 0.5
footnotesize = 8


def xy_scaler(X_train, X_test, y_train):
    """Scales the data as z = (x - u) / s. The scalers are fitted with the train sets.
    """
    
    # Defining the standarizers

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Fitting the scalers with the train sets
    
    if len(np.shape(X_train)) == 1:
        X_train = X_train.reshape(-1, 1)
    if len(np.shape(X_test)) == 1:
        X_test = X_test.reshape(-1, 1)
    X_scaler.fit(X_train)
    y_scaler.fit(y_train.reshape(-1, 1))

    # Standardizing the sets

    X_train = X_scaler.transform(X_train)
    X_test  = X_scaler.transform(X_test)
    y_train = np.ravel(y_scaler.transform(y_train.reshape(-1, 1)))
    return X_train, X_test, y_train, y_scaler


def y_descaler(y_list, y_scaler):
    """De-scales the data with the given scaler. A list of input data is de-scaled.
    """

    y_list_descaled = []
    for y_item in y_list:
            y_list_descaled.append(np.ravel(y_scaler.inverse_transform(np.array(y_item).reshape(-1, 1))))
    return y_list_descaled


def composition_concentration(structure):
    """Returns a list of strings: components of the formula (left) and their concentrations (right).
    It is indispensable that compounds start with capital letter.
    """

    composition   = []
    concentration = []

    components = re.findall('[A-Z][^A-Z]*', structure)
    for component in components:
        aux = re.split('(\d+)', component)
        composition.append(aux[0])
        if len(aux) > 1: concentration.append(aux[1])
        else:            concentration.append('1')
    return [' '.join(composition), ' '.join(concentration)]


def sign(coord_1, coord_2, coord_3):
    return (coord_1[0] - coord_3[0]) * (coord_2[1] - coord_3[1]) - (coord_2[0] - coord_3[0]) * (coord_1[1] - coord_3[1])


def plot_learning_curve(estimator, figure_name, X, y, axes=None, ylim=None, cv=None, n_jobs=mp.cpu_count(), dpi=400, scoring=None, train_sizes=np.linspace(0.1, 1.0, 5),):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores,   axis=1)
    test_scores_mean = np.mean(test_scores,   axis=1)
    test_scores_std = np.std(test_scores,     axis=1)
    fit_times_mean = np.mean(fit_times,       axis=1)
    fit_times_std = np.std(fit_times,         axis=1)

    # Plot learning curve
    
    fig = plt.figure(figsize=DL.get_figsize(1, 0.9))
    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color='r',
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color='g',
    )
    plt.plot(
        train_sizes, train_scores_mean, 'o-', color='r', label='Training score'
    )
    plt.plot(
        train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score'
    )
    plt.xlabel('Training examples', fontsize=footnotesize)
    plt.ylabel('Loss ($\mu\mathregular{m^{-1}}$)',              fontsize=footnotesize)
    plt.tick_params(axis='x', labelsize=footnotesize)
    plt.tick_params(axis='y', labelsize=footnotesize)
    plt.legend(loc='best', fontsize=footnotesize)
    plt.savefig(figure_name, dpi=dpi, bbox_inches='tight')
    plt.show()

    # Plot n_samples vs fit_times
    
    fig = plt.figure(figsize=DL.get_figsize(1, 0.9))
    plt.grid()
    plt.plot(train_sizes, fit_times_mean, 'o-')
    plt.fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    plt.xlabel('Training examples', fontsize=footnotesize)
    plt.ylabel('Fit times',         fontsize=footnotesize)
    plt.tick_params(axis='x', labelsize=footnotesize)
    plt.tick_params(axis='y', labelsize=footnotesize)
    #plt.legend(loc='best', fontsize=footnotesize)
    #plt.savefig(figure_name, dpi=dpi, bbox_inches='tight')
    plt.show()

    # Plot fit_time vs score
    
    fig = plt.figure(figsize=DL.get_figsize(1, 0.9))
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    plt.grid()
    plt.plot(fit_time_sorted, test_scores_mean_sorted, 'o-')
    plt.fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    plt.xlabel('Fit times', fontsize=footnotesize)
    plt.ylabel('Loss',      fontsize=footnotesize)
    plt.tick_params(axis='x', labelsize=footnotesize)
    plt.tick_params(axis='y', labelsize=footnotesize)
    #plt.legend(loc='best', fontsize=footnotesize)
    #plt.savefig(figure_name, dpi=dpi, bbox_inches='tight')
    plt.show()
    return train_sizes, train_scores, test_scores


class Limits:
    """Class to update and access the limits of each variable.
    """

    upper = None
    lower = None

    def _init_(self):
        self.upper = None
        self.lower = None


def generate_alloy_unitcell(substitutions_dict, path_to_alloy_folder, min_species_per_fu=1, n_atoms=None):
    """Create an unit-cell with the minimum number of atoms as to represent the solid solution. It checks all species in substitutions_dict with some non-zero concentration, and allows to tune the mininmum number of species per formula unit.
    
    Args:
        substitutions_dict (dict): concetration of each element to be considered within the solid solution.
        min_species_per_fu (int):  minimum number of atoms of an species per formula unit.
        n_atoms            (int):  number of atoms of an species per formula unit in the initial cell.
    """
    
    # Define the lowest concentration (except for 0), which drives the unit-cell generation
    min_concentration = 1
    for ref in substitutions_dict:
        for species in substitutions_dict[ref]:
            stc = substitutions_dict[ref][species]
            if (stc != 0) and (stc < min_concentration):
                min_concentration = stc
    
    # Compute the number of replications needed to have at least min_species_per_fu atoms of each species per formula unit
    total_replications = min_species_per_fu / min_concentration
    
    # Replications on each direction as cubic root of the total replications
    replication_factor = int(np.ceil(np.cbrt(total_replications)))
    print(f'Replication factor = {replication_factor}')
    
    # Save current working directory
    current_dir = os.getcwd()
    
    # Move to POSCAR folder
    os.chdir(path_to_alloy_folder)
    
    # Call vaspkit and get the unit-cell
    # POSCAR corresponds to the unit-cell of the pristine material
    command    = '~/vaspkit.1.3.5/bin/vaspkit'
    input_text = f'401\n1\n{replication_factor} {replication_factor} {replication_factor}'
    with open('/dev/null', 'w') as devnull:  # Avoid output generation
        subprocess.run(command, input=input_text.encode('utf-8'), shell=True, stdout=devnull, stderr=devnull)

    # Move the generated unit-cell to a POSCAR and clean
    os.system('mv SC* POSCAR')
    os.system('rm TRANSMAT')

    # Return to previous working directory
    os.chdir(current_dir)
    
    # If n_atoms is provided, prompt the number of atoms in the generated cell
    if n_atoms is not None: print(f'From {int(n_atoms)} to {int(np.power(replication_factor, 3) * n_atoms)} atoms')
