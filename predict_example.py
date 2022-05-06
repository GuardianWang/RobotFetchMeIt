import open3d as o3d
import numpy as np
import math
import torch
import os
import sys
import argparse
import importlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from model_util_sunrgbd import SunrgbdDatasetConfig
from ap_helper import parse_predictions

parser = argparse.ArgumentParser()
# ImVoteNet related options
parser.add_argument('--use_imvotenet', action='store_true', help='Use ImVoteNet (instead of VoteNet) with RGB.')
parser.add_argument('--max_imvote_per_pixel', type=int, default=3, help='Maximum number of image votes per pixel [default: 3]')
# Shared options with VoteNet
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
FLAGS = parser.parse_args()

if FLAGS.use_cls_nms:
    assert FLAGS.use_3d_nms

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]
if FLAGS.use_imvotenet:
    KEY_PREFIX_LIST = ['pc_img_']
    TOWER_WEIGHTS = {'pc_img_weight': 1.0}
else:
    KEY_PREFIX_LIST = ['pc_only_']
    TOWER_WEIGHTS = {'pc_only_weight': 1.0}


# Init the model and optimzier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

DATASET_CONFIG = SunrgbdDatasetConfig()
try:
    MODEL = importlib.import_module('votenet')
except Exception as e:
    print("model not imported")

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
               'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms, 'per_class_proposal': FLAGS.per_class_proposal,
               'conf_thresh': FLAGS.conf_thresh, 'dataset_config':DATASET_CONFIG}

USE_HEIGHT = True
NUM_POINTS = 20_000


def get_depth():
    # unit: mm
    depth = o3d.io.read_image("sample-image/depth.png")
    return depth


def get_pcd(to_np=True):
    depth = get_depth()
    height, width, *_ = np.asarray(depth).shape
    # TODO
    fov_y = 45  # top to bottom
    fx = fy = 0.5 * height / math.tan(math.radians(fov_y * 0.5))
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth,
        intrinsic=intrinsic,
        extrinsic=np.eye(4).astype(np.float32)
    )
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # to sunrgbd, rotate along x
    rot_euler = np.array([math.radians(90), 0, 0])
    rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(rot_euler)
    pcd.rotate(rot_mat, (0, 0, 0))

    return np.asarray(pcd.points) if to_np else pcd


def height_preprocess(pcd):
    # sunrgbd_detection_dataset.py
    floor_height = np.percentile(pcd[:, 2], 0.99)
    height = pcd[:, 2] - floor_height
    pcd = np.concatenate([pcd, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)
    return pcd


def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    pc_utils.py
    """
    if replace is None:
        replace = pc.shape[0] < num_sample
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def get_model_input():
    pcd = get_pcd()
    if USE_HEIGHT:
        pcd = height_preprocess(pcd)
    pcd = random_sampling(pcd, NUM_POINTS)
    return pcd[None, ...]


def viz_pcd(pcd):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)


def viz_full_pcd():
    pcd = get_pcd(to_np=False)
    viz_pcd(pcd)


def viz_model_input():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(get_model_input()[:, :3])
    viz_pcd(pcd)


def get_model():
    net = MODEL.VoteNet(num_class=DATASET_CONFIG.num_class,
                        num_heading_bin=DATASET_CONFIG.num_heading_bin,
                        num_size_cluster=DATASET_CONFIG.num_size_cluster,
                        mean_size_arr=DATASET_CONFIG.mean_size_arr,
                        num_proposal=FLAGS.num_target,
                        input_feature_dim=num_input_channel,
                        vote_factor=FLAGS.vote_factor,
                        sampling=FLAGS.cluster_sampling)
    if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print("Loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, epoch))
    return net


def predict(net, pcd):
    net.eval()  # set model to eval mode (for bn and dp)
    inputs = {'point_clouds': pcd}
    with torch.no_grad():
        end_points = net(inputs)
    end_points.update(inputs)
    parse_predictions(end_points, CONFIG_DICT, KEY_PREFIX_LIST[0])
    MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG, inference_switch=True, key_prefix=KEY_PREFIX_LIST[-1])
    
    for k, v in end_points.items():
        if isinstance(v, torch.Tensor):
            end_points[k] = v.detach().cpu().numpy()
        else:
            end_points[k] = np.array(v, dtype=object)
    np.savez("pred.npz", **end_points)


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


def get_pred_bbox(end_points, config, key_prefix, already_numpy=True):
    # dump_helper.py
    # INPUT

    # NETWORK OUTPUTS
    objectness_scores = end_points[key_prefix+'objectness_scores'] if already_numpy else end_points[key_prefix+'objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points[key_prefix+'center'] if already_numpy else end_points[key_prefix+'center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(torch.tensor(end_points[key_prefix+'heading_scores']), -1) if already_numpy else torch.argmax(end_points[key_prefix+'heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(torch.tensor(end_points[key_prefix+'heading_residuals']), 2, pred_heading_class.unsqueeze(-1)) if already_numpy else torch.gather(end_points[key_prefix+'heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class if already_numpy else pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).numpy() if already_numpy else pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(torch.tensor(end_points[key_prefix+'size_scores']), -1) if already_numpy else torch.argmax(end_points[key_prefix+'size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(torch.tensor(end_points[key_prefix+'size_residuals']), 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) if already_numpy else torch.gather(end_points[key_prefix+'size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).numpy() if already_numpy else pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    pred_mask = end_points[key_prefix+'pred_mask']  # B,num_proposal
    i = 0
    objectness_prob = softmax(objectness_scores[i, :, :])[:, 1]  # (K,)

    DUMP_CONF_THRESH = 0.5  # Dump boxes with obj prob larger than that.

    # Dump predicted bounding boxes
    if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
        num_proposal = pred_center.shape[1]
        obbs = []
        classes = []
        for j in range(num_proposal):
            obb = config.param2obb(pred_center[i, j, 0:3], pred_heading_class[i, j], pred_heading_residual[i, j],
                                   pred_size_class[i, j], pred_size_residual[i, j])
            # pred_size_class[i, j] is class
            # config.class2type[int(pred_size_class[i, j])] is class string
            classes.append(int(pred_size_class[i, j]))
            obbs.append(obb)
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_proposal, 7)
            selected = np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[i, :] == 1)
            confident_nms_obbs = obbs[selected]
            classes = list(map(lambda x: config.class2type[x], np.array(classes)[selected]))

            return confident_nms_obbs, classes
    return [], []


def parse_result():
    res = dict(np.load("pred.npz", allow_pickle=True))
    confident_nms_obbs, classes = get_pred_bbox(res, DATASET_CONFIG, key_prefix=KEY_PREFIX_LIST[-1], already_numpy=True)


def make_prediction():
    print("try to get point cloud")
    pcd = torch.tensor(get_model_input(), dtype=torch.float32).to(device)
    print("got point cloud")
    print("try to get network")
    net = get_model().to(device)
    print("got network")
    print("try to predict")
    predict(net, pcd)
    print("finish")


if __name__ == "__main__":
    # make_prediction()
    parse_result()
    pass
