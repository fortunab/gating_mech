import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from biogater.data import load_colorectal_histology
from biogater.models import build_biogater, build_convmixer, build_depthwise_only
from biogater.metrics import collect_labels_and_predictions, expected_calibration_error, extract_gate_values, gated_activation_diversity_score
from biogater.utils import set_seed, load_config, save_json, make_optimizer


def build_model(name, cfg):
    common = dict(
        input_shape=(cfg["img_size"], cfg["img_size"], 3),
        num_classes=cfg["num_classes"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        num_blocks=cfg["num_blocks"],
    )

    if name == "biogater":
        return build_biogater(
            **common,
            top_k=cfg["top_k"],
            l1_lambda=cfg["l1_lambda"],
        )
    if name == "convmixer":
        return build_convmixer(**common)
    if name == "depthwise":
        return build_depthwise_only(**common)

    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="biogater", choices=["biogater", "convmixer", "depthwise"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    output_dir = Path(args.output or f"outputs/models/{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = load_colorectal_histology(
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        train_size=cfg["train_size"],
        val_size=cfg["val_size"],
        test_size=cfg["test_size"],
    )

    model = build_model(args.model, cfg)
    model.compile(
        optimizer=make_optimizer(cfg["learning_rate"], cfg["decay_steps"], cfg["weight_decay"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        # Save in Keras format for portability.
        __import__("tensorflow").keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    y_true, y_prob = collect_labels_and_predictions(model, test_ds)
    ece = expected_calibration_error(y_true, y_prob)

    gates = extract_gate_values(model)
    gads = {name: gated_activation_diversity_score(vals) for name, vals in gates.items()}

    results = {
        "model": args.model,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "ece": float(ece),
        "gates": gates,
        "gads": gads,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }

    save_json(output_dir / "results.json", results)
    print("Saved results to:", output_dir / "results.json")


if __name__ == "__main__":
    main()
