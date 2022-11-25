import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from pathlib import Path
import datetime

from models.hdcam import HDCAM


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    bese_path = Path(cfg.model.path)


if __name__ == '__main__':
    pass
