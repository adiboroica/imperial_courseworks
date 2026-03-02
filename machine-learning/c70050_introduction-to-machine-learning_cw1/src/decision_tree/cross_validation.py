import numpy as np
from numpy.random import default_rng

from src.decision_tree.tree import decision_tree_learning, evaluate_tree, prune


def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits.
        n_instances (int): Number of instances to split.
        random_generator (np.random.Generator): A random generator.

    Returns:
        (list(np.ndarray)) : List of folds.
            Each element in the list is a numpy array giving the indices of the instances in that split.
    """

    # Generate a random permutation of indices from 0 to n_instances.
    shuffled_indices = random_generator.permutation(n_instances)
    # Split shuffled indices into almost equal sized splits.
    split_indices = np.array_split(shuffled_indices, n_splits)
    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        (list(np.ndarray)) : List of folds.
            Each element in the list is a tuple with two elements:
                - a numpy array containing the train indices
                - a numpy array containing the test indices
    """

    # Split the dataset into k splits.
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        test_indices = split_indices[k]
        train_indices = np.concatenate(split_indices[:k] + split_indices[k + 1:])
        folds.append([train_indices, test_indices])
    return folds


def cross_validation(dataset, n_folds=10, random_generator=default_rng()):
    """ Runs cross-validation on the given dataset.

    Args:
        dataset (np.ndarray): Dataset, numpy array with shape (N, K).
        n_folds (int) : Number of folds for cross-validation
        random_generator (np.random.Generator) : Random number generator for finding the k-folds.

    Output:
        (list[dict]) : a list of dictionaries consisting of (trained tree, depth, evaluation)
    """

    data_trees = []
    for (train_indices, test_indices) in train_test_k_fold(n_folds, len(dataset), random_generator):
        # Get the dataset from the correct splits.
        train_dataset = dataset[train_indices, :]
        test_dataset = dataset[test_indices, :]

        # Train the tree.
        (trained_tree, depth) = decision_tree_learning(train_dataset)
        # Evaluate the tree on the test dataset.
        evaluation = evaluate_tree(trained_tree, test_dataset)

        # Add the data of the tree to the list.
        tree_data = {
            "tree": trained_tree,
            "depth": depth,
            "evaluation": evaluation
        }
        data_trees.append(tree_data)

    return data_trees


def cross_validation_with_pruning(dataset, n_folds=10, random_generator=default_rng()):
    """ Runs cross-validation with pruning on the given dataset.

    Args:
        dataset (np.ndarray): Dataset, numpy array with shape (N, K).
        n_folds (int) : Number of folds for cross-validation.
        random_generator (np.random.Generator) : Random number generator for finding the k-folds.

    Output:
        (list[dict], list[dict]):
            - first dict contains the unpruned tree, depth and evaluation
            - second dict contains the pruned tree, depth and evaluation
    """

    data_unpruned_trees = []
    data_pruned_trees = []

    for (train_val_indices, test_indices) in train_test_k_fold(n_folds, len(dataset), random_generator):
        # Get the train-val and test datasets.
        train_val_dataset = dataset[train_val_indices, :]
        test_dataset = dataset[test_indices, :]

        # Train the tree on the train-val dataset, evaluate it on the test dataset and add it to the list.
        unpruned_tree, unpruned_depth = decision_tree_learning(train_val_dataset)
        unpruned_evaluation = evaluate_tree(unpruned_tree, test_dataset)
        unpruned_tree_data = {
            "tree": unpruned_tree,
            "depth": unpruned_depth,
            "evaluation": unpruned_evaluation,
        }
        data_unpruned_trees.append(unpruned_tree_data)

        for (train_indices, val_indices) in train_test_k_fold(n_folds - 1, len(train_val_dataset), random_generator):
            # Get the train and validation datasets.
            train_dataset = train_val_dataset[train_indices, :]
            val_dataset = train_val_dataset[val_indices, :]

            # Train the tree on the val dataset.
            trained_tree, _ = decision_tree_learning(train_dataset)

            # Prune the tree, evaluate it on the test dataset and add it to the list.
            pruned_tree, pruned_depth = prune(trained_tree, val_dataset)
            pruned_evaluation = evaluate_tree(pruned_tree, test_dataset)
            pruned_data = {
                "tree": pruned_tree,
                "depth": pruned_depth,
                "evaluation": pruned_evaluation
            }
            data_pruned_trees.append(pruned_data)

    return data_unpruned_trees, data_pruned_trees


def get_statistics(data_trees):
    """ Print statistics given a list of trees.

    Args:
        data_trees (list(dict)): List of dictionaries consisting of
            - trained tree
            - depth of the tree
            - evaluation of the tree

    Output:
        Return statistics using the given information.
    """

    accuracies = [data_tree["evaluation"]['accuracy'] for data_tree in data_trees]
    mean_accuracy = np.mean(np.array([data_tree["evaluation"]['accuracy'] for data_tree in data_trees]))

    depths = [data_tree['depth'] for data_tree in data_trees]
    mean_depth = np.mean(np.array([data_tree["depth"] for data_tree in data_trees]))

    mean_confusion = np.mean(np.array([data_tree["evaluation"]['confusion'] for data_tree in data_trees]), axis=0)

    mean_precision = np.mean(np.array([data_tree["evaluation"]['precision'] for data_tree in data_trees]), axis=0)
    mean_recall = np.mean(np.array([data_tree["evaluation"]['recall'] for data_tree in data_trees]), axis=0)
    mean_f1 = np.mean(np.array([data_tree["evaluation"]['f1_score'] for data_tree in data_trees]), axis=0)

    return {
        "accuracies": accuracies,
        "mean_accuracy": mean_accuracy,
        "depths": depths,
        "mean_depth": mean_depth,
        "mean_confusion": mean_confusion,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1
    }


def print_statistics(statistics):
    """ Print statistics given a list of trees.

    Args:
        statistics: dict consisting of statistics

    Output:
        Print statistics using the given information.
    """

    print("Accuracies:")
    print(statistics["accuracies"])
    print("Mean accuracy:")
    print(statistics["mean_accuracy"])

    print("Depths:")
    print(statistics["depths"])
    print("Mean depth:")
    print(statistics["mean_depth"])

    print("Mean confusion:")
    print(statistics["mean_confusion"])

    print("Mean precision on every class:")
    print(statistics["mean_precision"])
    print("Mean recall on every class:")
    print(statistics["mean_recall"])
    print("Mean f1 on every class:")
    print(statistics["mean_f1"])
