import numpy as np
import tensorflow as tf
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
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


def rect_and_lowpass(x, fpass):
    sr = 2048
    y = np.abs(x - np.mean(x, axis=-1))
    low_sos = signal.butter(1, fpass, btype='low', fs=sr, output='sos')
    filtered = signal.sosfilt(low_sos, y)
    return filtered


@tf.function
def mu_law(x, y):
    mu = 255.0
    encoded = tf.sign(x) * (tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu))
    return encoded, y


@tf.function
def frame_process(data, label, num_classes):
    frames = tf.transpose(tf.signal.frame(data, 640, 320, axis=1), perm=[1, 0, 2])
    label_vec = tf.one_hot(label, num_classes)
    label_vec = tf.broadcast_to(label_vec, shape=[tf.shape(frames)[0], num_classes])

    return frames, label_vec


if __name__ == '__main__':
    p = Path('/work/datasets/ninapro/')
    for path in p.glob('*_E1_*.mat'):
        try:
            f = scipy.io.loadmat(path)
            emg = np.transpose(f['emg'])
            labels = np.reshape(f['stimulus'], [np.shape(f['stimulus'])[0]])
            rep = np.reshape(f['repetition'], [np.shape(f['repetition'])[0]])
            exercise = f['exercise'][0][0]

            for i in range(1, 7):
                rep_xs = []
                rep_ys = []
                for j in range(1, 18):
                    rep_x, rep_y = frame_process(emg[:, (rep == i) & (labels == j)], j - 1, 17)
                    rep_xs.append(rep_x)
                    rep_ys.append(rep_y)
                    
                rep_xs = np.concatenate(rep_xs, axis=0)
                rep_ys = np.concatenate(rep_ys, axis=0)
                dataset = tf.data.Dataset.from_tensor_slices((rep_xs, rep_ys)).map(mu_law, num_parallel_calls=tf.data.AUTOTUNE)
                serialized_dataset = dataset.map(tf_serialize_example, num_parallel_calls=tf.data.AUTOTUNE)
                writer = tf.data.experimental.TFRecordWriter(str(p) + '/' + str(path.stem) + f'_rep{i}_ex{exercise}.tfrecord')
                writer.write(serialized_dataset)
                print(f'Done {str(path.stem)}_rep{str(i)}_ex{exercise}.tfrecord')

        except:
            print(f'Error at rep{i}, class{j}, {path.stem}')
