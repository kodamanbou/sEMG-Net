import numpy as np
import tensorflow as tf
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


@tf.function
def frame_process(data, label):
    frames = tf.transpose(tf.signal.frame(data, 64, 32), perm=[1, 0, 2])
    labels = tf.signal.frame(label, 64, 32)

    return frames, labels


if __name__ == '__main__':
    p = Path('/work/datasets/')
    for path in p.glob('*.mat'):
        infh = h5py.File(path, 'r')
        emg_flexors = np.array(infh['emg_flexors'], dtype=np.float32)
        time_step = emg_flexors.shape[2]
        emg_flexors = np.reshape(emg_flexors, (64, time_step))
        emg_extensors = np.array(infh['emg_extensors'], dtype=np.float32)
        emg_extensors = np.reshape(emg_extensors, (64, time_step))
        label = np.array(infh['class'], dtype=np.int8).squeeze(axis=0)
        infh.close()

        tr_x = np.concatenate([emg_flexors, emg_extensors], axis=0)
        tr_x_splits = np.array_split(tr_x, 8, axis=-1)
        label_splits = np.array_split(label, 8)
        for i, data in enumerate(tr_x_splits):
            emg_features, labels = frame_process(data, label_splits[i])
            dataset = tf.data.Dataset.from_tensor_slices((emg_features, labels))
            serialized_dataset = dataset.map(tf_serialize_example, num_parallel_calls=tf.data.AUTOTUNE)
            writer = tf.data.experimental.TFRecordWriter(str(p) + '/' + str(path.stem) + '_' + str(i + 1) + '.tfrecord')
            writer.write(serialized_dataset)

            print(f'Done {str(path.stem)}_{str(i + 1)}.tfrecord')
