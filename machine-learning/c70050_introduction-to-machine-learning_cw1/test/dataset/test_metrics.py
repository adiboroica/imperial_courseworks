import unittest

import numpy as np

from src.dataset import metrics


class TestMetrics(unittest.TestCase):
    GOLD_LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    PREDICTED_LABELS = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    CONFUSION_MATRIX = np.array([[3, 1], [2, 2]])

    def test_confusion_matrix(self):
        confusion = metrics.confusion_matrix(self.GOLD_LABELS, self.PREDICTED_LABELS)
        self.assertTrue((confusion == self.CONFUSION_MATRIX).all())

    def test_accuracy_from_confusion(self):
        acc = metrics.accuracy(self.GOLD_LABELS, self.PREDICTED_LABELS)
        self.assertEqual(acc, 0.625)

    def test_precision_from_confusion(self):
        precision, macro_precision = metrics.precision_from_confusion(self.CONFUSION_MATRIX)
        self.assertEqual(precision[0], 0.6)
        self.assertAlmostEqual(precision[1], 0.6666, places=3)
        self.assertAlmostEqual(macro_precision, 0.633, places=3)

    def test_recall_from_confusion(self):
        recall, macro_recall = metrics.recall_from_confusion(self.CONFUSION_MATRIX)
        self.assertEqual(recall[0], 0.75)
        self.assertEqual(recall[1], 0.5)
        self.assertEqual(macro_recall, 0.625)

    def test_f1_score(self):
        f1_scores, macro_f1 = metrics.f1_score_from_confusion(self.CONFUSION_MATRIX)
        self.assertAlmostEqual(f1_scores[0], 0.6666, places=3)
        self.assertAlmostEqual(f1_scores[1], 0.571, places=3)
        self.assertAlmostEqual(macro_f1, 0.619, places=3)
