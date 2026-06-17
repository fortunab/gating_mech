import argparse
from pathlib import Path
import sys
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from biogater.data import load_colorectal_histology
from biogater.metrics import collect_labels_and_predictions, expected_calibration_error
from biogater.utils import load_config, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, _, test_ds = load_colorectal_histology(
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        train_size=cfg["train_size"],
        val_size=cfg["val_size"],
        test_size=cfg["test_size"],
    )

    model = tf.keras.models.load_model(args.model_path, compile=False)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    loss, acc = model.evaluate(test_ds, verbose=1)
    y_true, y_prob = collect_labels_and_predictions(model, test_ds)
    ece = expected_calibration_error(y_true, y_prob)

    payload = {"loss": float(loss), "accuracy": float(acc), "ece": float(ece)}
    save_json(args.output, payload)
    print(payload)


if __name__ == "__main__":
    main()
