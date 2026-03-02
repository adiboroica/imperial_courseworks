import unittest

import numpy as np

from src.dataset.entropy import entropy, remainder, information_gain


class TestEntropy(unittest.TestCase):

    def test_entropy(self):
        self.assertAlmostEqual(entropy([1, 1]), 1, places=3)
        self.assertAlmostEqual(entropy([1, 1, 1, 1]), 2, places=3)

        self.assertAlmostEqual(entropy([97, 1, 1, 1]), 0.2419, places=3)

    def test_remainder(self):
        left_distribution = np.array([4, 2])
        right_distribution = np.array([3, 1])

        self.assertAlmostEqual(entropy(left_distribution), 0.918, places=3)
        self.assertAlmostEqual(entropy(right_distribution), 0.811, places=3)

        self.assertAlmostEqual(
            remainder(left_distribution, right_distribution), 0.8752, places=3
        )

    def test_information_gain_1(self):
        initial_distribution = np.array([5, 9])
        left_distribution = np.array([4, 3])
        right_distribution = np.array([1, 6])

        initial_entropy = entropy(initial_distribution)

        self.assertAlmostEqual(initial_entropy, 0.94, places=3)
        self.assertAlmostEqual(entropy(left_distribution), 0.985, places=3)
        self.assertAlmostEqual(entropy(right_distribution), 0.591, places=2)

        self.assertAlmostEqual(
            remainder(left_distribution, right_distribution), 0.788, places=3)

        self.assertAlmostEqual(
            information_gain(
                initial_entropy, left_distribution, right_distribution
            ),
            0.151,
            places=2
        )

    def test_information_gain_2(self):
        initial_distribution = np.array([5, 9])
        left_distribution = np.array([2, 6])
        right_distribution = np.array([3, 3])

        initial_entropy = entropy(initial_distribution)

        self.assertAlmostEqual(initial_entropy, 0.94, places=3)
        self.assertAlmostEqual(entropy(left_distribution), 0.811, places=3)
        self.assertAlmostEqual(entropy(right_distribution), 1, places=3)

        self.assertAlmostEqual(
            remainder(left_distribution, right_distribution), 0.892, places=3
        )

        self.assertAlmostEqual(
            information_gain(
                initial_entropy, left_distribution, right_distribution
            ),
            0.048,
            places=3
        )
