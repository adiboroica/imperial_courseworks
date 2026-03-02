import numpy as np

from src.dataset.entropy import entropy, information_gain
from src.dataset.metrics import metrics, accuracy, print_metrics


def predict_label(tree, value):
    """ Predict the label of the value using the given tree.

      Args:
          tree (dict) : The decision tree that has been generated.
          value: The values being compared to the decision tree for prediction.

      Returns:
          The predicted label using the given tree.
    """
    if tree['leaf']:
        return tree['label']
    if value[tree['attribute']] <= tree['value']:
        return predict_label(tree['left'], value)
    return predict_label(tree['right'], value)


def accuracy_of_tree(tree, test_dataset):
    """ Evaluates the tree on the test dataset and returns the accuracy.

    Args:
        tree (dict) : The decision tree that has been generated.
        test_dataset: Dataset used for calculating the accuracy

    Returns:
        Accuracy of the tree on the test dataset.
    """

    gold_labels = test_dataset[:, -1]
    predicted_labels = np.array(
        [predict_label(tree, value) for value in test_dataset[:, :-1]]
    )

    return accuracy(gold_labels, predicted_labels)


def evaluate_tree(tree, test_dataset):
    """ Evaluates the tree on the test dataset and returns the metrics.

    Args:
        tree (dict) : The decision tree that has been generated.
        test_dataset: Dataset used for calculating the metrics.

    Returns:
        (dict) Metrics of the tree using the given test dataset.
    """

    gold_labels = test_dataset[:, -1]
    predicted_labels = np.array(
        [predict_label(tree, value) for value in test_dataset[:, :-1]]
    )

    return metrics(gold_labels, predicted_labels)


def print_evaluation(evaluation):
    """ Prints the evaluation of the tree.

        Args:
            evaluation (dict): Dictionary of metrics about the tree

        Output:
            Prints the evaluation of the tree.
    """

    print_metrics(evaluation)


def find_split(dataset):
    """ Find the split with the highest information gain.

      Args:
        dataset (np.ndarray): shape (N, K)
                Initial dataset.

      Returns:
        (dict) : Information about the split point with the highest information gain.
            (info_gain, attribute, split_point)
    """

    dataset_size = dataset.shape[0]
    attributes_count = dataset.shape[1] - 1

    labels = dataset[:, -1]
    labels_split = np.bincount(labels.astype(int))

    # Compute here to avoid lots of recalculations.
    initial_entropy = entropy(labels_split)

    best_split = {
        'info_gain': -1,
        'split_point': 0,
        'attribute': 0,
    }
    for attribute_index in range(0, attributes_count):
        ordered_dataset = dataset[dataset[:, attribute_index].argsort()]

        # Decide split for attribute_index, so reset counts.
        left_count = np.zeros(np.shape(labels_split))
        right_count = np.copy(labels_split)

        i = 0
        while i < dataset_size:
            split_point = ordered_dataset[i, attribute_index]
            # Jump over the values that are repeated.
            while i < dataset_size and ordered_dataset[i, attribute_index] == split_point:
                left_count[int(ordered_dataset[i, -1])] += 1
                right_count[int(ordered_dataset[i, -1])] -= 1
                i += 1
            # Find the information gain.
            info_gain = information_gain(initial_entropy, left_count, right_count)
            # Update the best split point, if needed.
            if info_gain > best_split['info_gain']:
                best_split['info_gain'] = info_gain
                best_split['split_point'] = split_point
                best_split['attribute'] = attribute_index

    return best_split


def decision_tree_learning(training_dataset):
    """ Runs decision tree learning on the given dataset.

      Args:
          training_dataset (np.ndarray): Dataset, numpy array with shape (N, K).

      Returns:
          Root of the trained tree.
    """

    return decision_tree_learning_node(training_dataset, 0)


def decision_tree_learning_node(training_dataset, depth):
    """ Runs decision tree learning on the given dataset, given that the tree has been built until a certain depth.

      Args:
          training_dataset (np.ndarray): Dataset, numpy array with shape (N, K).
          depth (integer) : Current depth of the tree.

      Returns:
          (tuple) : New node of tree and maximum depth of the subtrees.
    """

    labels = training_dataset[:, -1]

    # If there is a single label in the whole training set, then there is no need to continue.
    if len(np.unique(labels)) == 1:
        leaf_node = {
            'leaf': True,
            'label': labels[0],
            'count': len(training_dataset)
        }
        return leaf_node, depth

    # Find the split with the highest information gain.
    split = find_split(training_dataset)

    # Split the training dataset.
    l_dataset = training_dataset[training_dataset[:, split['attribute']] <= split['split_point']]
    r_dataset = training_dataset[training_dataset[:, split['attribute']] > split['split_point']]

    # Recursively build the left and the right branch.
    l_branch, l_depth = decision_tree_learning_node(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning_node(r_dataset, depth + 1)

    # Return the split node and the depth.
    split_node = {
        'leaf': False,
        'attribute': split['attribute'],
        'value': split['split_point'],
        'left': l_branch,
        'right': r_branch,
    }
    return split_node, max(l_depth, r_depth)


def decide_prune(tree_node, validation_dataset):
    """ Decides if the current node needs to be pruned.

      Args:
          tree_node: Node directly connected to 2 leaves.
          validation_dataset: Dataset to validate the pruning on.

      Returns:
          (tuple): New node and new depth of the tree.
    """

    # Check that the left and right children are leaves.
    l_leaf, r_leaf = tree_node['left'], tree_node['right']
    assert l_leaf['leaf'] and r_leaf['leaf']

    # Construct the new tree node.
    new_label = l_leaf['label'] if l_leaf['count'] > r_leaf['count'] else r_leaf['label']
    new_count = l_leaf['count'] + r_leaf['count']
    new_tree_node = {
        'leaf': True,
        'label': new_label,
        'count': new_count
    }

    # If there is nothing to validate on, just prune the node.
    if len(validation_dataset) == 0:
        return new_tree_node, 0

    # Compute the accuracy before and after the pruning.
    before_acc = accuracy_of_tree(tree_node, validation_dataset)
    after_acc = accuracy_of_tree(new_tree_node, validation_dataset)

    # If there is an improvement, prune node. Otherwise, return the initial node.
    if after_acc >= before_acc:
        return new_tree_node, 0
    return tree_node, 1


def prune(tree_node, validation_dataset):
    """ Prunes the current tree node if there is an improvement on the validation dataset.

      Args:
          tree_node: The node to try the pruning on.
          validation_dataset: Dataset to validate pruning on.

      Returns:
          (tuple): New node and new depth of the tree.
    """

    # If the node is a leaf, then we can't prune.
    if tree_node['leaf']:
        return tree_node, 0

    # If node is not a leaf, it certainly has left and right children.
    left, right = tree_node['left'], tree_node['right']

    if left['leaf'] and right['leaf']:
        # Try to prune the current node.
        return decide_prune(tree_node, validation_dataset)
    else:
        # Recursively prune the left and the right branch.
        l_dataset = validation_dataset[validation_dataset[:, tree_node['attribute']] <= tree_node['value']]
        r_dataset = validation_dataset[validation_dataset[:, tree_node['attribute']] > tree_node['value']]
        left_pruned, l_depth = prune(left, l_dataset)
        right_pruned, r_depth = prune(right, r_dataset)

        # Update the value of the current node.
        pruned_node = {
            'leaf': False,
            'attribute': tree_node['attribute'],
            'value': tree_node['value'],
            'left': left_pruned,
            'right': right_pruned,
        }

        if left_pruned['leaf'] and right_pruned['leaf']:
            # Try to prune the new node.
            return decide_prune(pruned_node, validation_dataset)
        # Otherwise, return the new node.
        return pruned_node, max(l_depth, r_depth) + 1
