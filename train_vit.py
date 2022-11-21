import hydra
from omegaconf import DictConfig, OmegaConf
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

    return (parsed_element['emg'], parsed_element['label'])


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    base_path = Path(cfg.models.path)
    repetitions = [i for i in range(1, 6)]
    models = []
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    exp_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_1st_5Hz'

    for fold, (train_indices, valid_indices) in enumerate(kf.split(repetitions)):
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
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True
        )

        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=cfg.models.lr,
            weight_decay=cfg.models.weight_decay,
            beta_1=cfg.models.beta1,
            beta_2=cfg.models.beta2
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
        )

        train_paths = []
        for i in train_indices:
            for tr_path in base_path.glob('*_rep' + str(repetitions[i]) + '*.tfrecord'):
                train_paths.append(tr_path)

        train_raw_dataset = tf.data.TFRecordDataset(train_paths)
        train_raw_dataset = train_raw_dataset.shuffle(76)
        train_dataset = train_raw_dataset.map(
            deserialize_example, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg.models.batch_size)

        val_paths = []
        for i in valid_indices:
            for val_path in base_path.glob('*_rep' + str(repetitions[i]) + '*.tfrecord'):
                val_paths.append(val_path)

        val_raw_dataset = tf.data.TFRecordDataset(val_paths)
        val_raw_dataset = val_raw_dataset.shuffle(19)
        val_dataset = val_raw_dataset.map(
            deserialize_example, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg.models.batch_size)

        # start training
        log_dir = 'logs/' + exp_name + f'-fold{fold + 1}'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5)
        model.fit(train_dataset,
                  epochs=cfg.models.num_epochs,
                  validation_data=val_dataset,
                  callbacks=[tensorboard_callback,
                             early_callback])

        # save model
        model.save('outputs/vit_hgr_' + exp_name + f'_fold{fold + 1}')

        models.append(model)


if __name__ == '__main__':
    main()
