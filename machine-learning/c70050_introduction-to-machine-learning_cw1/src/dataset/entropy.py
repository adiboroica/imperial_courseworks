import numpy as np


def entropy(label_split):
    """ Compute the entropy for a given label split.

      Args:
          label_split (ndarray): shape(K, ),
                Labels bincounts for dataset

      Returns:
          entropy: The entropy of the dataset using the last column.
    """

    sample_count = np.sum(label_split)

    if sample_count == 0:
        return 0
    probabilities = label_split / sample_count
    probabilities = probabilities[probabilities > 0]
    return np.sum(-probabilities * np.log2(probabilities))


def remainder(l_label_split, r_label_split):
    """ Compute the average entropy of the produced subsets.

    Args:
        l_label_split (np.ndarray): shape (N1, K1)
              Left label split
        r_label_split (np.ndarray): shape (N2, K2)
              Right label split

    Returns:
        remainder: The average entropy of the produced subsets.
    """
    l_entropy = entropy(l_label_split)
    r_entropy = entropy(r_label_split)

    l_count = np.sum(l_label_split)
    r_count = np.sum(r_label_split)
    p = l_count / (l_count + r_count)

    return p * l_entropy + (1 - p) * r_entropy


def information_gain(dataset_entropy, left_count, right_count):
    """ Compute the information gained by splitting the dataset.

      Args:
          dataset_entropy (float): Initial entropy of the dataset
          left_count (ndarray): shape(K1, ),
                Labels bincounts for the left dataset
          right_count (ndarray): shape(K2, ),
                Labels bincounts for the right dataset

      Returns:
          information_gain (float): The information gain of this split.
    """

    return dataset_entropy - remainder(left_count, right_count)
