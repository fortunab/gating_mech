import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_example(example, img_size: int = 75):
    image = tf.image.resize(example["image"], [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    label = example["label"]
    return image, label


def load_colorectal_histology(
    img_size: int = 75,
    batch_size: int = 64,
    seed: int = 42,
    train_size: int = 4000,
    val_size: int = 500,
    test_size: int = 500,
):
    """Load TensorFlow colorectal_histology and create train/val/test splits.

    The dataset has 5000 images. The default split follows the notebook:
    4000 train, 500 validation, 500 test.
    """
    ds = tfds.load("colorectal_histology", split="train", as_supervised=False)

    ds = (
        ds.shuffle(5000, seed=seed, reshuffle_each_iteration=True)
        .map(lambda x: preprocess_example(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    )

    train_ds = ds.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = ds.skip(train_size).take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = ds.skip(train_size + val_size).take(test_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def load_cifar10(
    img_size: int = 75,
    batch_size: int = 64,
    seed: int = 42,
):
    """Optional (for adv.)."""
    def preprocess(example):
        image = tf.image.resize(example["image"], [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, example["label"]

    train = tfds.load("cifar10", split="train", as_supervised=False)
    train = train.shuffle(50000, seed=seed, reshuffle_each_iteration=True).map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train.take(45000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = train.skip(45000).take(5000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tfds.load("cifar10", split="test", as_supervised=False)
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
