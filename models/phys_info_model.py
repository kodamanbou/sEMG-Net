import tensorflow as tf


class FeatureExtractNet(tf.keras.Model):
    def __init__(self, filters=128, timesteps=128, muscle_forces=5, joints=5) -> None:
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.drop3 = tf.keras.layers.Dropout(0.3)
        self.out = tf.keras.layers.Dense(muscle_forces + joints)
        
    def call(self, inputs, training=False):
        paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
        x = tf.pad(inputs, paddings, 'CONSTANT')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.dense2(x)
        x = self.bn3(x)
        x = self.drop3(x)

        y = self.out(x)

        return y
