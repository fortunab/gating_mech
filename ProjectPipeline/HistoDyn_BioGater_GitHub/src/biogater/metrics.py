import numpy as np
import tensorflow as tf


def expected_calibration_error(y_true, y_prob, n_bins: int = 15):
    """Expected Calibration Error for multiclass probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.mean(mask) * abs(bin_acc - bin_conf)

    return float(ece)


def collect_labels_and_predictions(model, dataset):
    y_true, y_prob = [], []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)
        y_true.append(labels.numpy())

    return np.concatenate(y_true), np.concatenate(y_prob)


def gated_activation_diversity_score(gate_values, eps: float = 1e-8):
    """GADS: entropy over normalized gate usage.

    Higher values mean more diverse use of operations.
    Lower values mean specialization around fewer operations.
    """
    gates = np.asarray(gate_values, dtype=np.float64)

    if gates.ndim == 1:
        avg = gates
    else:
        normalized = gates / (np.sum(gates, axis=1, keepdims=True) + eps)
        avg = np.mean(normalized, axis=0)

    p = np.abs(avg)
    p = p / (np.sum(p) + eps)
    return float(-np.sum(p * np.log(p + eps)))


def extract_gate_values(model):
    """Extract BioGater gate vectors from a Keras model."""
    values = {}

    for layer in model.layers:
        if hasattr(layer, "gates"):
            values[layer.name] = layer.gates.numpy().tolist()

    return values
