import os
import sys
import argparse
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'TextCondRobotFetch/pointnet'))
from TextCondRobotFetch.pointnet.inference import get_text_model, inference, add_shape_arguments

parser = argparse.ArgumentParser()
add_shape_arguments(parser)
FLAGS = parser.parse_args()


def monitor_bbox_folder(txt_path="selected_bbox_folder_path.txt", result_file="result.txt")
    while True:
    	  time.sleep(0.1)
    	  if os.path.exists(txt_path):
            with open(txt_path, "w") as f:
                line = f.read().strip()
            result_path = os.path.join(line, result_file)
            if os.path.exists(result_path):
                continue
        break
    return line
        
if __name__ == "__main__":
    shape_model = get_text_model(FLAGS)
    monitor_bbox_folder()
    pass
