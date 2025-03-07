import tensorflow as tf

class DropBlock1D(tf.keras.layers.Layer):
    def __init__(self, keep_prob=0.9, block_size=7, beta=0.9):
        super(DropBlock1D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.beta = beta

    def normalize(self, input):
        min_c = tf.reduce_min(input, axis=1, keepdims=True)
        max_c = tf.reduce_max(input, axis=1, keepdims=True)
        input_norm = (input - min_c) / (max_c - min_c + 1e-8)
        return input_norm

    def call(self, input, training=None):
        if not training or self.keep_prob == 1:
            return input

        gamma = (1. - self.keep_prob) / self.block_size
        for sh in input.shape[1:]:
            gamma *= sh / (sh - self.block_size + 1)

        M = tf.cast(tf.random.uniform(tf.shape(input)) < gamma, input.dtype)
        Msum = tf.nn.conv1d(tf.expand_dims(M, axis=1),
                            tf.ones((1, 1, self.block_size), dtype=input.dtype),
                            stride=1,
                            padding='SAME')

        Msum = tf.cast(Msum < 1, input.dtype)
        input2 = input * Msum
        x_norm = self.normalize(input2)
        mask = tf.cast(x_norm > self.beta, input.dtype)
        block_mask = 1 - (mask * x_norm)
        return input * block_mask
