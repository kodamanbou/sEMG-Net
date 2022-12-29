import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from sklearn.model_selection import KFold
from pathlib import Path
import datetime

from models.hdcam import HDCAM


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

    base_path = Path(cfg.model.path)
    exp_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    repetitions = [i for i in range(1, 7)]
    models = []
    kf = KFold(n_splits=6, shuffle=True, random_state=10)

    for fold, (train_indices, valid_indices) in enumerate(kf.split(repetitions)):
        print(f'Fold {fold + 1}')
        model = HDCAM(
            cfg.model.num_classes,
            cfg.model.num_channels,
            cfg.model.num_splits,
            cfg.model.num_heads
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg.model.lr,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.model.lr
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
                print(train_paths)

        train_raw_dataset = tf.data.TFRecordDataset(train_paths)
        train_raw_dataset = train_raw_dataset.shuffle(600)
        train_dataset = train_raw_dataset.map(
            deserialize_example, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg.model.batch_size)

        val_paths = []
        for i in valid_indices:
            for val_path in base_path.glob('*_rep' + str(repetitions[i]) + '*.tfrecord'):
                val_paths.append(val_path)

        val_raw_dataset = tf.data.TFRecordDataset(val_paths)
        val_raw_dataset = val_raw_dataset.shuffle(120)
        val_dataset = val_raw_dataset.map(
            deserialize_example, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg.model.batch_size)

        # start training
        log_dir = 'logs/' + exp_name + f'-fold{fold + 1}'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5)
        model.fit(train_dataset,
                  epochs=cfg.model.num_epochs,
                  validation_data=val_dataset,
                  callbacks=[tensorboard_callback,
                             early_callback])

        # save model
        model.save('outputs/hdcam_' + exp_name + f'_fold{fold + 1}')

        models.append(model)

if __name__ == '__main__':
    main()
