import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy import signal
from pathlib import Path
import datetime

from models.vit_hgr import VisionTransformer


def deserialize_example(example_proto):
    feature_description = {
        'emg': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_element = tf.io.parse_single_example(
        example_proto, feature_description)
    parsed_element['emg'] = tf.io.parse_tensor(
        parsed_element['emg'], out_type=tf.float32)
    parsed_element['label'] = tf.io.parse_tensor(
        parsed_element['label'], out_type=tf.float32)

    return parsed_element


@tf.function
def mu_law(input):
    x = input['emg']
    mu = 255.0
    encoded = tf.sign(x) * (tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu))
    return encoded, input['label']


@tf.function
def butter_lowpass(x, gpass, gstop, fs, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [gpass/nyq, gstop/nyq], btype='bandstop')
    y = signal.filtfilt(b, a, x)
    return y


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    base_path = Path(cfg.models.path)
    participants = np.expand_dims(
        np.array(cfg.models.participants, dtype=object), axis=-1)
    models = []
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    for fold, (train_indices, valid_indices) in enumerate(kf.split(participants)):
        print(f'Fold {fold + 1}')
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

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg.models.lr,
            decay_steps=200000,
            decay_rate=0.96,
            staircase=True
        )

        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr_schedule,
            weight_decay=cfg.models.weight_decay,
            beta_1=cfg.models.beta1,
            beta_2=cfg.models.beta2
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy(),
                     tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')]
        )

        train_paths = []
        for i in train_indices:
            for tr_path in base_path.glob(str(participants[i]) + '*.tfrecord'):
                train_paths.append(tr_path)

        train_raw_dataset = tf.data.TFRecordDataset(train_paths)
        train_raw_dataset = train_raw_dataset.shuffle(64)
        train_dataset = train_raw_dataset.map(
            deserialize_example, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(
            mu_law, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg.models.batch_size)

        val_paths = []
        for i in valid_indices:
            for val_path in base_path.glob(str(participants[i]) + '*.tfrecord'):
                val_paths.append(val_path)

        val_raw_dataset = tf.data.TFRecordDataset(val_paths)
        val_raw_dataset = val_raw_dataset.shuffle(16)
        val_dataset = val_raw_dataset.map(
            deserialize_example, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(
            mu_law, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg.models.batch_size)

        # start training
        log_dir = 'logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + \
            f'-fold{fold + 1}'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_crossentropy', patience=3)
        model.fit(train_dataset,
                  epochs=cfg.models.num_epochs,
                  validation_data=val_dataset,
                  callbacks=[tensorboard_callback, early_callback])

        # save model
        model.save('outputs/vit_hgr_' +
                   datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') +
                   f'_fold{fold + 1}')

        models.append(model)


if __name__ == '__main__':
    main()
