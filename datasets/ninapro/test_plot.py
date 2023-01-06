import numpy as np
import tensorflow as tf
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path


def rect_and_lowpass(x, fpass):
    sr = 2048
    y = np.abs(x - np.mean(x, axis=-1))
    low_sos = signal.butter(1, fpass, btype='low', fs=sr, output='sos')
    filtered = signal.sosfilt(low_sos, y)
    return filtered


@tf.function
def mu_law(x):
    mu = 255.0
    encoded = tf.sign(x) * (tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu))
    return encoded


if __name__ == '__main__':
    p = Path('/work/datasets/ninapro/')
    for path in p.glob('*.mat'):
        f = scipy.io.loadmat(path)
        emg = np.transpose(f['emg'])
        label = np.reshape(f['stimulus'], [np.shape(f['stimulus'])[0]])
        rep = np.reshape(f['repetition'], [np.shape(f['repetition'])[0]])
        channel = 0

        emgs = emg[channel, 2048*50:2048*110]
        reps = rep[2048*50:2048*110]
        labels = label[2048*50:2048*110]
        
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(321, xlabel='time[s]')
        time_axis = np.linspace(0, reps.shape[0] / 2048, reps.shape[0])
        ax1.set_xlabel('time[s]')
        ax1.plot(time_axis, emgs, label=f'raw emg[{channel + 1}]')
        ax1.legend()

        spec = np.fft.rfft(emgs)
        freqs = np.fft.rfftfreq(len(emgs)) * 2048
        ax2 = fig.add_subplot(322)
        ax2.loglog(freqs, spec)
        ax2.set_xlabel('(raw)Frequency in Hertz[Hz]')
        ax2.set_ylabel('Frequency Domain (Spectrum) Magnitude')

        rectified = np.array(rect_and_lowpass(emgs, fpass=5), dtype=np.float32)
        ax3 = fig.add_subplot(323, xlabel='time[s]')
        ax3.set_xlabel('time[s]')
        ax3.plot(time_axis, rectified, label=f'rectified emg[{channel + 1}]')
        ax3.legend()

        spec = np.fft.rfft(rectified)
        freqs = np.fft.rfftfreq(len(rectified)) * 2048
        ax4 = fig.add_subplot(324)
        ax4.loglog(freqs, spec)
        ax4.set_xlabel('(rectified)Frequency in Hertz[Hz]')
        ax4.set_ylabel('Frequency Domain (Spectrum) Magnitude')

        normalized = mu_law(rectified).numpy()
        ax5 = fig.add_subplot(325, xlabel='time[s]')
        time_axis = np.linspace(0, reps.shape[0] / 2048, reps.shape[0])
        ax5.set_xlabel('time[s]')
        # ax5.plot(time_axis, normalized, label=f'mu_law emg[{channel + 1}]')
        ax5.plot(time_axis, reps, label='repetition')
        ax5.plot(time_axis, labels, label='stimulus')
        ax5.legend()

        spec = np.fft.rfft(normalized)
        freqs = np.fft.rfftfreq(len(normalized)) * 2048
        ax6 = fig.add_subplot(326)
        ax6.loglog(freqs, spec)
        ax6.set_xlabel('(mu_law)Frequency in Hertz[Hz]')
        ax6.set_ylabel('Frequency Domain (Spectrum) Magnitude')
        
        fig.tight_layout()
        fig.savefig(f'{path.stem}.png')
        plt.close()
