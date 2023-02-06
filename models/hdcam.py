import tensorflow as tf
from typing import List


class HDConvEncoder(tf.keras.layers.Layer):
    def __init__(self, num_splits: int):
        super(HDConvEncoder, self).__init__()
        self.num_splits = num_splits
        self.enc_layers = [
            tf.keras.layers.DepthwiseConv1D(
                kernel_size=3, padding='same', data_format='channels_first')
            for _ in range(num_splits)
        ]

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pwconv = tf.keras.layers.DepthwiseConv1D(
            kernel_size=1, data_format='channels_first', activation=tf.keras.activations.gelu)

    def call(self, inputs):
        x = tf.split(inputs, num_or_size_splits=self.num_splits, axis=1)
        y = []
        y.append(self.enc_layers[0](x[0]))

        for x_i, layer in zip(x[1:], self.enc_layers[1:]):
            y.append(layer(x_i + y[-1]))

        y = tf.concat(y, axis=1)
        y = self.layernorm(y)
        y = self.pwconv(y)
        return inputs + y


class MHSAttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, dropout=0.1):
        super(MHSAttentionEncoder, self).__init__()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=64, dropout=dropout)
        self.pwconv = tf.keras.layers.DepthwiseConv1D(
            kernel_size=1, data_format='channels_first', activation=tf.keras.activations.gelu)

    def call(self, inputs, training):
        x = inputs
        y = self.layernorm1(x)
        y = self.att(y, y, training=training)
        y = self.layernorm2(x + y)
        y = self.pwconv(y)
        return x + y


class HDCAM(tf.keras.Model):
    def __init__(
        self,
        num_classes: int,
        num_channels: List[int] = [24, 32, 64],
        num_splits: List[int] = [3, 4, 4],
        num_heads: List[int] = [4, 4],
        dropout=0.1
    ):
        super(HDCAM, self).__init__()

        # stage 1
        self.stem = tf.keras.layers.Conv1D(
            num_channels[0], kernel_size=10, strides=10, data_format='channels_first')
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.hdconv1 = HDConvEncoder(num_splits[0])

        # stage 2
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.downsample1 = tf.keras.layers.Conv1D(
            num_channels[1], kernel_size=2, strides=2, data_format='channels_first')
        self.hdconv2 = [
            HDConvEncoder(num_splits[1])
            for _ in range(2)
        ]
        self.mhs_enc1 = MHSAttentionEncoder(num_heads[0], dropout=dropout)

        # stage 3
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.downsample2 = tf.keras.layers.Conv1D(
            num_channels[2], kernel_size=2, strides=2, data_format='channels_first')
        self.hdconv3 = [
            HDConvEncoder(num_splits[2])
            for _ in range(4)
        ]
        self.mhs_enc2 = MHSAttentionEncoder(num_heads[1], dropout=dropout)

        # final stage
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D(
            data_format='channels_first')
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training):
        x = self.stem(inputs)
        x = self.layernorm1(x)
        x = self.hdconv1(x, training=training)

        x = self.layernorm2(x)
        x = self.downsample1(x)

        for layer in self.hdconv2:
            x = layer(x, training=training)

        x = self.mhs_enc1(x, training=training)

        x = self.layernorm3(x)
        x = self.downsample2(x)

        for layer in self.hdconv3:
            x = layer(x, training=training)

        x = self.mhs_enc2(x, training=training)

        x = self.avgpool(x)
        logits = self.dense(x)
        return logits
