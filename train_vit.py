import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf


@tf.function
def mu_law(x, mu):
    return tf.sign(x) * (tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu))


@tf.function
def train():
    print('Training loop.')


@hydra.main(config_path='conf')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    main()
