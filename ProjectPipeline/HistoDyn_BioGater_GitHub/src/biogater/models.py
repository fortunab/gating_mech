import tensorflow as tf
from tensorflow.keras import layers, models


class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="same",
        )
        self.flatten = layers.Reshape((-1, embed_dim))

    def call(self, x):
        return self.flatten(self.projection(x))

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
        return config


def cheap_ops_dict(embed_dim: int):
    """Biologically inspired cheap operations.

    These operations approximate interpretable histology transformations:
    stain normalization, contrast amplification, polarity/edge behavior,
    discretization, and simple texture shifts.
    """
    return {
        "identity": layers.Lambda(lambda x: x),
        "softmax_gate": layers.Lambda(lambda x: x * tf.nn.softmax(x, axis=1)),
        "zeros": layers.Lambda(lambda x: tf.zeros_like(x)),
        "noise_injection": layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), stddev=0.01)),
        "spatial_dropout": layers.SpatialDropout1D(0.1),
        "scale_small": layers.Lambda(lambda x: 0.1 * x),
        "scale_large": layers.Lambda(lambda x: 10.0 * x),
        "negate": layers.Lambda(lambda x: -x),
        "abs": layers.Lambda(lambda x: tf.abs(x)),
        "square": layers.Lambda(lambda x: tf.square(x)),
        "sqrt": layers.Lambda(lambda x: tf.sqrt(tf.abs(x) + 1e-5)),
        "sign": layers.Lambda(lambda x: tf.sign(x)),
        "binarize": layers.Lambda(lambda x: tf.cast(x > 0.0, tf.float32)),
        "tanh_gate": layers.Lambda(lambda x: x * tf.tanh(x)),
        "sigmoid_gate": layers.Lambda(lambda x: x * tf.nn.sigmoid(x)),
        "mean_subtract": layers.Lambda(lambda x: x - tf.reduce_mean(x, axis=1, keepdims=True)),
        "std_norm": layers.Lambda(
            lambda x: (x - tf.reduce_mean(x, axis=1, keepdims=True))
            / (tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-5)
        ),
        "global_context": layers.Lambda(
            lambda x: tf.repeat(
                tf.reduce_mean(x, axis=1, keepdims=True),
                repeats=tf.shape(x)[1],
                axis=1,
            )
        ),
        "reverse": layers.Lambda(lambda x: tf.reverse(x, axis=[1])),
        "roll": layers.Lambda(lambda x: tf.roll(x, shift=1, axis=1)),
        "linear_skip": models.Sequential(
            [layers.Dense(embed_dim), layers.Activation("linear")],
            name="linear_skip_op",
        ),
    }


class MultiHeadCheapOpBlockWeighted(layers.Layer):
    """BioGater block with learnable operation gates and optional top-k behavior.

    During training, all operations are softly weighted to keep gradients stable.
    During inference, only top-k gates are retained, approximating conditional sparse execution.
    """
    def __init__(
        self,
        embed_dim: int,
        op_names=None,
        top_k: int = 4,
        l1_lambda: float = 0.01,
        projection: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.op_names = op_names or [
            "std_norm",
            "scale_large",
            "sign",
            "binarize",
            "mean_subtract",
            "abs",
            "reverse",
            "global_context",
        ]
        self.top_k = top_k
        self.l1_lambda = l1_lambda
        self.projection = projection
        self.pre_ln = layers.LayerNormalization(epsilon=1e-6)
        self.ops = cheap_ops_dict(embed_dim)
        self.project = layers.Dense(embed_dim) if projection else layers.Activation("linear")

    def build(self, input_shape):
        self.gates = self.add_weight(
            name="operation_gates",
            shape=(len(self.op_names),),
            initializer="ones",
            trainable=True,
            regularizer=tf.keras.regularizers.L1(self.l1_lambda),
        )
        super().build(input_shape)

    def call(self, x, training=None, return_gates=False):
        h = self.pre_ln(x)
        outputs = []

        for op_name in self.op_names:
            op = self.ops[op_name]
            y = op(h)

            # Some experimental ops can change channel count. Resize to embed_dim.
            if y.shape[-1] != self.embed_dim:
                y = layers.Dense(self.embed_dim)(y)

            outputs.append(y)

        stacked = tf.stack(outputs, axis=0)  # [ops, batch, tokens, dim]

        if training:
            weights = tf.nn.softmax(self.gates)
        else:
            k = tf.minimum(self.top_k, tf.shape(self.gates)[0])
            _, indices = tf.math.top_k(self.gates, k=k)
            mask = tf.reduce_sum(tf.one_hot(indices, depth=tf.shape(self.gates)[0]), axis=0)
            masked = self.gates * mask
            weights = masked / (tf.reduce_sum(masked) + 1e-8)

        weights = tf.reshape(weights, (-1, 1, 1, 1))
        mixed = tf.reduce_sum(stacked * weights, axis=0)
        mixed = self.project(mixed)
        out = x + mixed

        if return_gates:
            return out, self.gates
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "op_names": self.op_names,
                "top_k": self.top_k,
                "l1_lambda": self.l1_lambda,
                "projection": self.projection,
            }
        )
        return config


class ConvMixerBlock(layers.Layer):
    def __init__(self, embed_dim: int, kernel_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.dwconv = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same", activation="gelu")
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.pwconv = layers.Conv2D(embed_dim, kernel_size=1, padding="same")
        self.activation = layers.Activation("gelu")

    def call(self, x):
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]
        d = tf.shape(x)[2]
        h_side = tf.cast(tf.sqrt(tf.cast(n, tf.float32)), tf.int32)

        h = self.ln1(x)
        h2d = tf.reshape(h, (b, h_side, h_side, d))
        h2d = self.dwconv(h2d)
        h2d = tf.reshape(h2d, (b, n, d))
        x = x + h2d

        h = self.ln2(x)
        h2d = tf.reshape(h, (b, h_side, h_side, d))
        h2d = self.pwconv(h2d)
        h2d = self.activation(h2d)
        return tf.reshape(h2d, (b, n, d))


class DepthwiseOnlyBlock(layers.Layer):
    def __init__(self, embed_dim: int, kernel_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.dwconv = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same", activation="gelu")

    def call(self, x):
        b = tf.shape(x)[0]
        n = tf.shape(x)[1]
        d = tf.shape(x)[2]
        h_side = tf.cast(tf.sqrt(tf.cast(n, tf.float32)), tf.int32)

        h = self.ln(x)
        h2d = tf.reshape(h, (b, h_side, h_side, d))
        h2d = self.dwconv(h2d)
        h2d = tf.reshape(h2d, (b, n, d))
        return x + h2d


def build_biogater(
    input_shape=(75, 75, 3),
    num_classes: int = 8,
    patch_size: int = 5,
    embed_dim: int = 64,
    num_blocks: int = 2,
    top_k: int = 4,
    l1_lambda: float = 0.01,
):
    inputs = layers.Input(shape=input_shape)
    x = PatchEmbedding(patch_size, embed_dim)(inputs)

    for i in range(num_blocks):
        x = MultiHeadCheapOpBlockWeighted(
            embed_dim=embed_dim,
            top_k=top_k,
            l1_lambda=l1_lambda,
            name=f"biogater_block_{i+1}",
        )(x)

    x_avg = layers.GlobalAveragePooling1D()(x)
    x_max = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([x_avg, x_max])
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="HistoDyn_BioGater")


def build_convmixer(
    input_shape=(75, 75, 3),
    num_classes: int = 8,
    patch_size: int = 5,
    embed_dim: int = 48,
    num_blocks: int = 2,
):
    inputs = layers.Input(shape=input_shape)
    x = PatchEmbedding(patch_size, embed_dim)(inputs)

    for i in range(num_blocks):
        x = ConvMixerBlock(embed_dim, name=f"convmixer_block_{i+1}")(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="ConvMixer")


def build_depthwise_only(
    input_shape=(75, 75, 3),
    num_classes: int = 8,
    patch_size: int = 5,
    embed_dim: int = 96,
    num_blocks: int = 2,
):
    inputs = layers.Input(shape=input_shape)
    x = PatchEmbedding(patch_size, embed_dim)(inputs)

    for i in range(num_blocks):
        x = DepthwiseOnlyBlock(embed_dim, name=f"depthwise_block_{i+1}")(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="DepthwiseOnly")
