import os
import sys
import argparse
import time
import glob
import numpy as np
from random import sample


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'TextCondRobotFetch/pointnet'))
from TextCondRobotFetch.pointnet.inference import get_text_model, inference, add_shape_arguments

parser = argparse.ArgumentParser()
add_shape_arguments(parser)
FLAGS = parser.parse_args()


def monitor_bbox_folder(txt_path="selected_bbox_folder_path.txt", result_file="result.txt"):
    while True:
        time.sleep(0.1)
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                line = f.read().strip()
            result_path = os.path.join(line, result_file)
            if os.path.exists(result_path):
                continue
        else:
            continue
        break
    return line
        
        
def wait_until_can_read(selected_bbox_folder):
    prev = 0
    curr = 0
    while curr == 0 or prev != curr:
        time.sleep(1)
        prev, curr = curr, len(os.listdir(selected_bbox_folder))
    return curr


def pred_shape(selected_bbox_folder, model, partial_scans, latent_folder="TextCondRobotFetch/embeddings", latent_fmt="shape_{:04d}.npy",
               latent_id=0, result_file="result.txt", npy_fmt="{:03d}.npy"):
    result_file_path = os.path.join(selected_bbox_folder, result_file)
    latent_path = os.path.join(latent_folder, latent_fmt.format(latent_id))
    n = len(os.listdir(selected_bbox_folder))
    latent = np.load(latent_path)
    preds = []
    with open(result_file_path, 'a') as f:
        for i in range(n):
            pcd_path = os.path.join(selected_bbox_folder, npy_fmt.format(i))
            pcd = np.load(pcd_path)
            res = inference(pcd, latent, model, FLAGS, sample(partial_scans, 2))
            preds.append(res)
            f.write('1' if res else '0')
    return preds


def get_partial_scans(folder="../subdataset", fmt="**/*_partial.npz"):
    p = os.path.join(folder, fmt)
    files = glob.glob(p, recursive=True)

    return files


if __name__ == "__main__":
    print("loading shape model")
    shape_model = get_text_model(FLAGS)
    print("scanning partial point clouds")
    partial_scans = get_partial_scans()
    while True:
        print("monitoring bbox folder")
        selected_bbox_folder = monitor_bbox_folder()
        print("waiting bbox folder")
        n_bbox = wait_until_can_read(selected_bbox_folder)
        print("got {} bboxes".format(n_bbox))
        pred_shape(selected_bbox_folder, shape_model, partial_scans)
    pass
