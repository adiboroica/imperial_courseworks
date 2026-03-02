import unittest

import numpy as np

from part1_nn_lib import Preprocessor


class TestPreprocessor(unittest.TestCase):

    DATASET = np.array([[0, 1], [2, 3]])
    PREPROCESSOR = Preprocessor(DATASET)

    def test_apply(self):
        preprocessed_dataset = self.PREPROCESSOR.apply(self.DATASET)
        expected_preprocessed_dataset = np.array([[0, 0], [1, 1]])

        self.assertTrue(
            (preprocessed_dataset == expected_preprocessed_dataset).all()
        )

    def test_unapply(self):
        preprocessed_dataset = self.PREPROCESSOR.apply(self.DATASET)
        reverted_dataset = self.PREPROCESSOR.revert(preprocessed_dataset)

        self.assertTrue(
            (reverted_dataset == self.DATASET).all()
        )
