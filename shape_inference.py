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
        
        
def wait_until_can_read(selected_bbox_folder):
	 prev = 0
	 curr = len(os.listdir(selected_bbox_folder))
    while curr != 0 and prev != curr:
    	 time.sleep(1)
    	 prev, curr = curr, len(os.listdir(selected_bbox_folder))
    return curr
	
	
if __name__ == "__main__":
    shape_model = get_text_model(FLAGS)
    print("monitoring bbox folder")
    selected_bbox_folder = monitor_bbox_folder()
    print("waiting bbox folder")
    n_bbox = wait_until_can_read(selected_bbox_folder)
    print("got {} bboxes".format(n_bbox))
    pass
