import tensorflow as tf
import numpy as np
from typing import List


class HDConvEncoder(tf.keras.layers.Layer):
    def __init__(self, num_splits: int):
        super(HDConvEncoder, self).__init__()
        self.num_splits = num_splits
        self.dwconvs = [
            tf.keras.layers.DepthwiseConv1D(kernel_size=3, padding='same', data_format='channels_first')
            for _ in range(num_splits)
        ]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pwconv = tf.keras.layers.DepthwiseConv1D(
            kernel_size=1, data_format='channels_first', activation=tf.keras.activations.gelu)

    def call(self, inputs, *args, **kwargs):
        x = tf.split(inputs, num_or_size_splits=self.num_splits, axis=1)
        y = []
        y.append(self.dwconvs[0](x[0]))

        for x_i, layer in zip(x[1:], self.dwconvs[1:]):
            y.append(layer(x_i + y[-1]))

        y = tf.concat(y, axis=1)
        y = self.layernorm(y)
        y = self.pwconv(y)
        return inputs + y


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class MHSAttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, num_channels: int, num_heads: int):
        super(MHSAttentionEncoder, self).__init__()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.att = MultiHeadSelfAttention(num_channels, num_heads)
        self.pwconv = tf.keras.layers.DepthwiseConv1D(
            kernel_size=1, data_format='channels_first', activation=tf.keras.activations.gelu)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        y = self.layernorm1(x)
        y = self.att(y)
        y = self.layernorm2(x + y)
        y = self.pwconv(y)
        return x + y


class HDCAM(tf.keras.Model):
    def __init__(
        self,
        num_classes: int,
        num_channels: List[int]=[24, 32, 64],
        num_splits: List[int]=[3, 4, 4],
        num_heads: List[int]=[4, 4],
    ):
        super(HDCAM, self).__init__()
        
        # stage 1
        self.stem = tf.keras.layers.Conv1D(num_channels[0], kernel_size=10, strides=10, data_format='channels_first')
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.hdconv1 = HDConvEncoder(num_splits[0])
        
        # stage 2
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.downsample1 = tf.keras.layers.Conv1D(num_channels[1], kernel_size=2, strides=2, data_format='channels_first')
        self.hdconv2 = [
            HDConvEncoder(num_splits[1])
            for _ in range(2)
        ]
        self.mhs_enc1 = MHSAttentionEncoder(32, num_heads[0])

        #stage 3
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.downsample2 = tf.keras.layers.Conv1D(num_channels[2], kernel_size=2, strides=2, data_format='channels_first')
        self.hdconv3 = [
            HDConvEncoder(num_splits[2])
            for _ in range(4)
        ]
        self.mhs_enc2 = MHSAttentionEncoder(16, num_heads[1])

        # final stage
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training):
        x = self.stem(inputs)
        x = self.layernorm1(x)
        x = self.hdconv1(x)

        x = self.layernorm2(x)
        x = self.downsample1(x)

        for layer in self.hdconv2:
            x = layer(x)

        x = self.mhs_enc1(x)

        x = self.layernorm3(x)
        x = self.downsample2(x)

        for layer in self.hdconv3:
            x = layer(x)

        x = self.mhs_enc2(x)

        x = self.avgpool(x)
        logits = self.dense(x)
        return logits
