import tensorflow as tf
from pathlib import Path
import argparse


def main(path: Path):
    model = tf.keras.models.load_model(path)
    pass


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_path')
    main(Path('/work/outputs/hdcam_hdcam_2023_01_07_09_41_15_fold1/'))
