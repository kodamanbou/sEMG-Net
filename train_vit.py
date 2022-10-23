import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import tensorflow as tf
from scipy import signal
from pathlib import Path

from models.vit_hgr import VisionTransformer


def deserialize_example(example_proto):
    feature_description = {
        'emg': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_element = tf.io.parse_single_example(example_proto, feature_description)
    parsed_element['emg'] = tf.io.parse_tensor(parsed_element['emg'], out_type=tf.float32)
    parsed_element['label'] = tf.io.parse_tensor(parsed_element['label'], out_type=tf.int8)

    return parsed_element


@tf.function
def mu_law(x, mu):
    return tf.sign(x) * (tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu))


@tf.function
def butter_lowpass(x, gpass, gstop, fs, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [gpass/nyq, gstop/nyq], btype='bandstop')
    y = signal.filtfilt(b, a, x)
    return y


@tf.function
def train():
    print('Training loop.')


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model = VisionTransformer(
        image_size=cfg.models.image_size,
        patch_size=cfg.models.patch_size,
        num_layers=cfg.models.num_layers,
        num_classes=cfg.models.num_classes,
        d_model=cfg.models.d_model,
        num_heads=cfg.models.num_heads,
        mlp_dim=cfg.models.mlp_dim,
        channels=cfg.models.channels
    )

    model.build(input_shape=(None, 64, 64, 1))
    model.summary()

    base_path = Path(cfg.models.path)
    rng = np.random.default_rng()
    tr_parts = rng.choice(cfg.models.participants, 4, replace=False)
    print(f'Training participants: {tr_parts}')
    tr_paths = []
    for part in tr_parts:
        for tr_path in base_path.glob(part + '*.tfrecord'):
            tr_paths.append(tr_path)

    raw_dataset = tf.data.TFRecordDataset(tr_paths)
    deserialized_dataest = raw_dataset.map(deserialize_example)

    for record in deserialized_dataest.take(1):
        print(repr(record))


if __name__ == '__main__':
    main()
