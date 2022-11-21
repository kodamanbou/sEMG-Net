import numpy as np
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import h5py


def rect_and_lowpass(x, fpass):
    sr = 2048
    y = np.abs(x - np.reshape(np.mean(x, axis=-1), (64, 1)))
    low_sos = signal.butter(1, fpass, btype='low', fs=sr, output='sos')
    filtered = signal.sosfilt(low_sos, y)
    return filtered


@tf.function
def mu_law(x):
    mu = 255.0
    encoded = tf.sign(x) * (tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu))
    return encoded


@tf.function
def frame_process(data):
    frames = tf.transpose(tf.signal.frame(data, 64, 32, axis=1), perm=[1, 0, 2, 3])
    return frames


if __name__ == '__main__':
    sr = 2048
    channel = 30
    infh = h5py.File('/work/datasets/s3.mat', 'r')
    emg_flexors = np.array(infh['emg_flexors'], dtype=np.float32)
    time_step = emg_flexors.shape[2]
    emg_flexors = np.reshape(emg_flexors, (64, time_step))
    emg_extensors = np.array(infh['emg_extensors'], dtype=np.float32)
    emg_extensors = np.reshape(emg_extensors, (64, time_step))
    ext_filtered = np.array(rect_and_lowpass(emg_extensors, fpass=5), dtype=np.float32)
    labels = np.array(infh['class'], dtype=np.int64).squeeze(axis=0)
    repetition = np.array(infh['repetition'], dtype=np.int64).squeeze(axis=0)
    infh.close()

    tr_x = []
    ex_x = []
    reps = []
    classes = []
    for i in range(1, 6):
        for label in range(1, 3):
            tr_x.append(emg_extensors[channel, (repetition == i) & (labels == label)])
            ex_x.append(ext_filtered[channel, (repetition == i) & (labels == label)])
            reps.append(repetition[(repetition == i) & (labels == label)])
            classes.append(labels[(repetition == i) & (labels == label)])

    tr_x = np.concatenate(tr_x)
    ex_x = np.concatenate(ex_x)
    reps = np.concatenate(reps)
    classes = np.concatenate(classes)

    fig = plt.figure(figsize=(12, 8))
    time_axis = np.linspace(0, reps.shape[0] / sr, reps.shape[0])
    ax1 = fig.add_subplot(221, xlabel='time[s]')
    ax1.plot(time_axis, tr_x, label=f'raw extensor[{channel}]')
    ax1.legend()

    spec = np.fft.rfft(tr_x)
    freqs = np.fft.rfftfreq(len(tr_x)) * 2048
    ax2 = fig.add_subplot(222)
    ax2.loglog(freqs, spec)
    ax2.set_xlabel('(raw)Frequency in Hertz[Hz]')
    ax2.set_ylabel('Frequency Domain (Spectrum) Magnitude')

    ax3 = fig.add_subplot(223, xlabel='time[s]')
    ax3.plot(time_axis, mu_law(ex_x).numpy(), label=f'processed extensor[{channel}]')
    ax3.legend()

    spec = np.fft.rfft(ex_x)
    freqs = np.fft.rfftfreq(len(ex_x)) * 2048
    ax4 = fig.add_subplot(224)
    ax4.loglog(freqs, spec)
    ax4.set_xlabel('(proc)Frequency in Hertz[Hz]')
    ax4.set_ylabel('Frequency Domain (Spectrum) Magnitude')

    fig.savefig('output.png')
