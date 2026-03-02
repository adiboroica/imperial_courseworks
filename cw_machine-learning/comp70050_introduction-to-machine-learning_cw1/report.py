import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import default_rng

from main import CLEAN_DATA_PATH, NOISY_DATA_PATH
from src.decision_tree.cross_validation import cross_validation, get_statistics, cross_validation_with_pruning


def generate_clean_cross_val_image():
    dataset = np.array(np.loadtxt(CLEAN_DATA_PATH))
    seed = 60020

    data_trees = cross_validation(dataset, n_folds=10, random_generator=default_rng(seed))
    statistics = get_statistics(data_trees)

    dataframe = pd.DataFrame(statistics["mean_confusion"])

    # Save figure for trees and then reset.
    sns.set()
    sns.heatmap(dataframe, annot=True, cmap="rocket_r", fmt=".2f")
    plt.savefig('clean_cross-val.png')
    sns.reset_defaults()
    plt.clf()


def generate_clean_cross_val_with_pruning_image():
    dataset = np.array(np.loadtxt(CLEAN_DATA_PATH))
    seed = 60020

    data_unpruned_trees, data_pruned_trees = \
        cross_validation_with_pruning(dataset, n_folds=10, random_generator=default_rng(seed))

    statistics_unpruned_trees = get_statistics(data_unpruned_trees)
    statistics_pruned_trees = get_statistics(data_pruned_trees)

    dataframe_unpruned_trees = pd.DataFrame(statistics_unpruned_trees["mean_confusion"])
    dataframe_pruned_trees = pd.DataFrame(statistics_pruned_trees["mean_confusion"])

    # Save figure for unpruned trees and then reset.
    sns.set()
    sns.heatmap(dataframe_unpruned_trees, annot=True, cmap="rocket_r", fmt=".2f")
    plt.savefig('clean_cross-val-with-pruning_unpruned.png')
    sns.reset_defaults()
    plt.clf()

    # Save figure for pruned trees and then reset.
    sns.set()
    sns.heatmap(dataframe_pruned_trees, annot=True, cmap="rocket_r", fmt=".2f")
    plt.savefig('clean_cross-val-with-pruning_pruned.png')
    sns.reset_defaults()
    plt.clf()


def generate_noisy_cross_val_with_pruning_image():
    dataset = np.array(np.loadtxt(NOISY_DATA_PATH))
    seed = 60012

    data_unpruned_trees, data_pruned_trees = \
        cross_validation_with_pruning(dataset, n_folds=10, random_generator=default_rng(seed))

    statistics_unpruned_trees = get_statistics(data_unpruned_trees)
    statistics_pruned_trees = get_statistics(data_pruned_trees)

    dataframe_unpruned_trees = pd.DataFrame(statistics_unpruned_trees["mean_confusion"])
    dataframe_pruned_trees = pd.DataFrame(statistics_pruned_trees["mean_confusion"])

    # Save figure for unpruned trees and then reset.
    sns.set()
    sns.heatmap(dataframe_unpruned_trees, annot=True, cmap="rocket_r", fmt=".2f")
    plt.savefig('noisy_cross-val-with-pruning_unpruned.png')
    sns.reset_defaults()
    plt.clf()

    # Save figure for pruned trees and then reset.
    sns.set()
    sns.heatmap(dataframe_pruned_trees, annot=True, cmap="rocket_r", fmt=".2f")
    plt.savefig('noisy_cross-val-with-pruning_pruned.png')
    sns.reset_defaults()
    plt.clf()


def main():
    generate_clean_cross_val_image()
    generate_clean_cross_val_with_pruning_image()
    generate_noisy_cross_val_with_pruning_image()


if __name__ == "__main__":
    main()
