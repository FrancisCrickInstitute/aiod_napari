import numpy as np
from skimage.metrics import hausdorff_distance

# TODO: Switch to Dask for calculation


def dice(masks1: np.ndarray, masks2: np.ndarray) -> float:
    """Dice coefficient, a quotient of similarity in the range [0, 1]. 1 is perfect overlap, 0 is no overlap."""
    intersection = np.sum(np.logical_and(masks1, masks2))
    return 2 * intersection / (np.sum(masks1) + np.sum(masks2))


def iou(masks1: np.ndarray, masks2: np.ndarray) -> float:
    """Intersection over union, or Jaccard index, measures the overlap between two masks. Correlated with Dice, but slightly harsher on mistakes."""
    intersection = np.sum(np.logical_and(masks1, masks2))
    union = np.sum(np.logical_or(masks1, masks2))
    return intersection / union


def precision(preds: np.ndarray, labels: np.ndarray) -> float:
    """Precision, or positive predictive value, measures the proportion of predicted positives that are true positives."""
    tp = np.sum(np.logical_and(preds, labels))
    fp = np.sum(np.logical_and(preds == 1, labels == 0))
    return tp / (tp + fp)


def recall(preds: np.ndarray, labels: np.ndarray) -> float:
    """Recall, or sensitivity, measures the proportion of true positives that are predicted positives."""
    tp = np.sum(np.logical_and(preds, labels))
    fn = np.sum(np.logical_and(preds == 0, labels == 1))
    return tp / (tp + fn)


def hausdorff_dist(
    masks1: np.ndarray, masks2: np.ndarray, method: str = "standard"
) -> float:
    """Hausdorff distance between two masks."""
    return hausdorff_distance(masks1, masks2, method=method)


def labelled_to_binary(masks: np.ndarray) -> np.ndarray:
    # Convert labelled instance masks to a flat binary mask
    return (masks > 0).astype(int)
