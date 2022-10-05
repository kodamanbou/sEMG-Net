import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from scipy import signal

from models.vit_hgr import VisionTransformer


@tf.function
def mu_law(x, mu):
    return tf.sign(x) * (tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu))


@tf.function
def butter_lowpass(x, lowcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
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


if __name__ == '__main__':
    main()
