import os
import sys
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'TextCondRobotFetch/pointnet'))
from TextCondRobotFetch.pointnet.inference import get_text_model, inference, add_shape_arguments

parser = argparse.ArgumentParser()
add_shape_arguments(parser)
FLAGS = parser.parse_args()


if __name__ == "__main__":
    get_text_model(FLAGS)
    pass
