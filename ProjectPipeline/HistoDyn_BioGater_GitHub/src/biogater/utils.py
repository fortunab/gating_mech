from pathlib import Path
import json
import yaml
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def make_optimizer(initial_lr: float = 1e-3, decay_steps: int = 10000, weight_decay: float = 1e-4):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=0.0,
    )
    return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
