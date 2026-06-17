import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/models/biogater/results.json")
    parser.add_argument("--output", default="outputs/figures/gates.png")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    gates = results.get("gates", {})
    if not gates:
        raise RuntimeError("No gates found. Train BioGater first.")

    # Plot first gate vector.
    name, values = next(iter(gates.items()))
    labels = [f"op_{i}" for i in range(len(values))]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Gate value")
    plt.title(f"Gate values: {name}")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
