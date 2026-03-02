import unittest

from numpy.random import default_rng

from src.decision_tree.cross_validation import k_fold_split, train_test_k_fold


def pairwise_disjoint(list_of_arrays):
    list_of_sets = list(set(array) for array in list_of_arrays)
    no_sets = len(list_of_sets)

    for i in range(0, no_sets):
        for j in range(i + 1, no_sets):
            if len(list_of_sets[i].intersection(list_of_sets[j])) != 0:
                return False

    return True


class TestCrossValidation(unittest.TestCase):
    SEED = 60012
    rand_gen = default_rng(SEED)

    def test_k_fold_split(self):
        n_splits = 3
        n_instances = 20
        splits = k_fold_split(n_splits, n_instances, self.rand_gen)

        for split in splits:
            # Every split need to have length between int(n_instances/n_splits) and int(n_instances/n_splits) + 1.
            self.assertTrue(int(n_instances / n_splits) <= len(split) <= int(n_instances / n_splits) + 1)

        # The splits need to be pairwise disjoint.
        self.assertTrue(pairwise_disjoint(splits))

        # The union of the splits needs to be {0, 1, ..., n_instances - 1}.
        # (this also checks that all the indices are >= 0 and < n_instances)
        self.assertTrue(set().union(*splits) == set(range(0, n_instances)))

    def test_train_test_k_fold(self):
        n_splits = 4
        n_instances = 30
        folds = train_test_k_fold(n_splits, n_instances, self.rand_gen)

        for (train_indices, test_indices) in folds:
            union_of_indices = set().union(train_indices, test_indices)

            # The union of the splits needs to be {0, 1, ..., n_instances - 1}.
            # (this also checks that all the indices are >= 0 and < n_instances)
            self.assertTrue(union_of_indices == set(range(0, n_instances)))

            # The splits need to be pairwise disjoint.
            self.assertTrue(pairwise_disjoint([train_indices, test_indices]))

            # Test indices split needs to contain only one fold.
            self.assertTrue(int(n_instances / n_splits) <= len(test_indices) <= int(n_instances / n_splits) + 1)
