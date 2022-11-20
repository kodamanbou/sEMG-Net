import numpy as np
import tensorflow as tf
from scipy import signal
import h5py
from pathlib import Path


def serialize_example(emg_feature, label):
    serialized_emg = tf.io.serialize_tensor(emg_feature)
    serialized_label = tf.io.serialize_tensor(label)
    feature = {
        'emg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_emg.numpy()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_label.numpy()]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(emg_feature, label):
    tf_string = tf.py_function(
        serialize_example,
        (emg_feature, label),
        tf.string
    )
    return tf_string


def lowpass(x, fpass):
    # fpass should be in range of 5-100[Hz]
    sr = 2048
    y = np.abs(x - np.reshape(np.mean(x, axis=-1), (8, 8, 1)))
    sos = signal.butter(1, fpass, btype='low', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, y)
    return filtered


@tf.function
def mu_law(x, y):
    mu = 255.0
    encoded = tf.sign(x) * (tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu))
    return encoded, y


@tf.function
def frame_process(data, label, num_classes):
    frames = tf.transpose(tf.signal.frame(data, 64, 32, axis=2), perm=[2, 0, 1, 3, 4])
    frame_num = tf.shape(frames)[0]
    frames = tf.reshape(frames, shape=[frame_num, 8, 8, -1])
    label_vec = tf.one_hot(label, num_classes)
    label_vec = tf.broadcast_to(label_vec, shape=[tf.shape(frames)[0], num_classes])

    return frames, label_vec


if __name__ == '__main__':
    p = Path('/work/datasets/')
    for path in p.glob('*.mat'):
        infh = h5py.File(path, 'r')
        emg_flexors = np.array(infh['emg_flexors'], dtype=np.float32)
        emg_flexors = np.array(lowpass(emg_flexors, fpass=5), dtype=np.float32)
        time_step = emg_flexors.shape[2]
        emg_flexors = np.expand_dims(emg_flexors, axis=-1)
        emg_extensors = np.array(infh['emg_extensors'], dtype=np.float32)
        emg_extensors = np.array(lowpass(emg_extensors, fpass=5), dtype=np.float32)
        emg_extensors = np.expand_dims(emg_extensors, axis=-1)
        labels = np.array(infh['class'], dtype=np.int64).squeeze(axis=0)
        repetition = np.array(infh['repetition'], dtype=np.int64).squeeze(axis=0)
        infh.close()

        tr_x = np.concatenate([emg_flexors, emg_extensors], axis=-1)
        for i in range(1, 6):
            rep_x = []
            rep_y = []
            for label in range(1, 66):
                tr_x_splits = tr_x[:, :, (repetition == i) & (labels == label), :]
                tr_x_splits, tr_y = frame_process(tr_x_splits, label - 1, 65)
                rep_x.append(tr_x_splits)
                rep_y.append(tr_y)

            rep_x = np.concatenate(rep_x, axis=0)
            rep_y = np.concatenate(rep_y, axis=0)
            dataset = tf.data.Dataset.from_tensor_slices((rep_x, rep_y)).map(mu_law, num_parallel_calls=tf.data.AUTOTUNE)
            serialized_dataset = dataset.map(tf_serialize_example, num_parallel_calls=tf.data.AUTOTUNE)
            writer = tf.data.experimental.TFRecordWriter(str(p) + '/' + str(path.stem) + '_rep' + str(i) + '.tfrecord')
            writer.write(serialized_dataset)
            print(f'Done {str(path.stem)}_rep{str(i)}.tfrecord')
