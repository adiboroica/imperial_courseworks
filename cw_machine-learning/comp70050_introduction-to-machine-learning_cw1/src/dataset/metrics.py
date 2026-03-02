import numpy as np


def metrics(gold_labels, predicted_labels):
    """ Compute the metrics given the gold labels and predicted labels.

    Args:
        gold_labels (np.ndarray): (np.ndarray): shape (N, )
                    The correct ground truth/gold standard labels
        predicted_labels (np.ndarray): shape (N, )
                    The predicted labels

    Returns:
        (dict) : Metrics of the prediction.
    """

    assert len(gold_labels) == len(predicted_labels)

    confusion = confusion_matrix(gold_labels, predicted_labels)
    acc = accuracy(gold_labels, predicted_labels)
    precision, macro_precision = precision_from_confusion(confusion)
    recall, macro_recall = recall_from_confusion(confusion)
    f1_score, macro_f1_score = f1_score_from_confusion(confusion)

    return {
        "confusion": confusion,
        "accuracy": acc,
        "precision": precision,
        "macro_precision": macro_precision,
        "recall": recall,
        "macro_recall": macro_recall,
        "f1_score": f1_score,
        "macro_f1_score": macro_f1_score,
    }


def print_metrics(metrics):
    """ Prints the given metrics.

        Args:
            metrics (dict): Dictionary of metrics.

        Output:
            Prints the given metrics.
    """

    print("Confusion:")
    print(metrics["confusion"])

    print("Accuracy:")
    print(metrics["accuracy"])

    print("Precision:")
    print(metrics["precision"])
    print("Macro precision:")
    print(metrics["macro_precision"])

    print("Recall:")
    print(metrics["recall"])
    print("Macro recall:")
    print(metrics["macro_recall"])

    print("F1 Score:")
    print(metrics["f1_score"])
    print("Macro F1 Score:")
    print(metrics["macro_f1_score"])


def confusion_matrix(gold_labels, predicted_labels, class_labels=None):
    """ Compute the confusion matrix.

    Args:
        gold_labels (np.ndarray): (np.ndarray): shape (N, )
                    The correct ground truth/gold standard labels
        predicted_labels (np.ndarray): shape (N, )
                    The predicted labels
        class_labels (np.ndarray): a list of unique class labels. 
                    Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes. 
                   Rows are ground truth per class, columns are predictions
    """

    assert len(gold_labels) == len(predicted_labels)

    # If no class_labels are given, we obtain the set of unique class labels from
    # the union of the correct labels and predicted labels.
    if not class_labels:
        class_labels = np.unique(np.concatenate(
            (gold_labels, predicted_labels)
        ))

    class_labels_indexes = {}

    for index in range(0, len(class_labels)):
        class_labels_indexes.update({class_labels[index]: index})

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    for index in range(0, len(gold_labels)):
        gold_label_index = class_labels_indexes[gold_labels[index]]
        predicted_label_index = class_labels_indexes[predicted_labels[index]]
        confusion[gold_label_index][predicted_label_index] += 1

    return confusion


def accuracy(gold_labels, predicted_labels):
    """ Compute the accuracy of the predictions.

    Args:
        gold_labels (np.ndarray): (np.ndarray): shape (N, )
            The correct ground truth/gold standard labels.
        predicted_labels (np.ndarray): shape (N, )
            The predicted labels.

    Returns:
        The accuracy of the predictions.
    """

    assert len(gold_labels) == len(predicted_labels)

    if len(gold_labels) == 0:
        return 0

    return np.sum(predicted_labels == gold_labels) / len(gold_labels)


def precision_from_confusion(confusion):
    """ Compute the precision score per class given the confusion matrix.

    Also return the macro-averaged precision across classes.

    Args:
        confusion (np.ndarray): shape (C, C) confusion matrix where C is the number of classes.
                    The rows are ground truth per class, columns are predictions.

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the 
              precision for class c
            - macro-precision is macro-averaged precision (a float) 
    """

    # Compute the precision per class
    correct_predictions = np.diag(confusion)
    all_class_predictions = np.sum(confusion, axis=0)
    precision = correct_predictions / all_class_predictions

    # Compute the macro-averaged precision
    macro_precision = np.mean(precision) if len(precision) != 0 else 0

    return precision, macro_precision


def recall_from_confusion(confusion):
    """ Compute the recall score per class given the ground truth and predictions

    Also return the macro-averaged recall across classes.

    Args:
        confusion (np.ndarray): shape (C, C) confusion matrix where C is the number of classes.
                    The rows are ground truth per class, columns are predictions.

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the 
                recall for class c
            - macro-recall is macro-averaged recall (a float) 
    """

    # Compute the recall per class
    correct_predictions = np.diag(confusion)
    actual_class_count = np.sum(confusion, axis=1)
    # TODO: what if actual_class_count has 0 elements
    recall = correct_predictions / actual_class_count

    # Compute the macro-averaged recall
    macro_recall = np.mean(recall) if len(recall) != 0 else 0

    return recall, macro_recall


def f1_score_from_confusion(confusion):
    """ Compute the F1-score per class given the confusion matrix

    Also return the macro-averaged F1-score across classes.

    Args:
        confusion (np.ndarray): shape (C, C) confusion matrix where C is the number of classes.
                    The rows are ground truth per class, columns are predictions.

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the 
              f1-score for class c
            - macro_f1 is macro-averaged f1-score (a float)
    """

    (precisions, _) = precision_from_confusion(confusion)
    (recalls, _) = recall_from_confusion(confusion)

    # Compute the per-class F1
    f1_scores = np.zeros((len(precisions),))
    for index in range(0, len(precisions)):
        p = precisions[index]
        r = recalls[index]
        f1_scores[index] = (2 * p * r) / (p + r) if p + r != 0 else 0

    # Compute the macro-averaged F1
    macro_f1 = np.mean(f1_scores) if len(f1_scores) != 0 else 0

    return f1_scores, macro_f1
