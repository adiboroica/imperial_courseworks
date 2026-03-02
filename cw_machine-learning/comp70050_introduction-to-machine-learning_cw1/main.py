from src.decision_tree.tree import *
from src.decision_tree.cross_validation import *
from src.decision_tree.visualisation import *

CLEAN_DATA_PATH = "wifi_db/clean_dataset.txt"
NOISY_DATA_PATH = "wifi_db/noisy_dataset.txt"

DATASET_PATH = CLEAN_DATA_PATH


def main():
    dataset = np.array(np.loadtxt(DATASET_PATH))

    # Decision Tree Learning on the whole dataset.
    decision_tree_learning_tree, depth = decision_tree_learning(dataset)
    # Visualise the tree.
    visualise_tree(decision_tree_learning_tree, name="test", max_depth=depth)

    # Decision Tree Learning on a part of the dataset.
    train_indices, test_indices = train_test_k_fold(10, dataset.shape[0])[0]  # pick one fold
    train_dataset = dataset[train_indices, :]
    test_dataset = dataset[test_indices, :]
    tree, _ = decision_tree_learning(train_dataset)
    # Evaluate the tree on the test dataset.
    evaluation = evaluate_tree(tree, test_dataset)
    print("\"\"\"")
    print("   Performance of decision_tree_learning()")
    print("\"\"\"")
    print_evaluation(evaluation)
    print("")
    print("")

    # Cross Validation
    data_trees_cross_validation = cross_validation(dataset)
    # Print statistics of the trained trees.
    print("\"\"\"")
    print("   Performance of cross_validation()")
    print("\"\"\"")
    print_statistics(get_statistics(data_trees_cross_validation))
    print("")
    print("")

    # Cross Validation with Pruning.
    data_unpruned_trees, data_pruned_trees = cross_validation_with_pruning(dataset)
    # Print statistics of the unpruned trees.
    print("\"\"\"")
    print("   Performance of cross-validation_with_pruning() - on unpruned trees")
    print("\"\"\"")
    print_statistics(get_statistics(data_unpruned_trees))
    print("")
    print("")
    # Print statistics of the pruned trees.
    print("\"\"\"")
    print("   Performance of cross_validation_with_pruning() - on pruned trees")
    print("\"\"\"")
    print_statistics(get_statistics(data_pruned_trees))
    print("")
    print("")


if __name__ == "__main__":
    main()
