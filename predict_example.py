import open3d as o3d
import numpy as np
import math
import torch
import os
from copy import deepcopy
import sys
import argparse
import importlib
import time
import cv2
from scipy import ndimage

import bosdyn.client
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder, block_until_arm_arrives, \
    block_for_trajectory_cmd
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, get_a_tform_b
from bosdyn.client.docking import blocking_undock, blocking_dock_robot
from bosdyn.client import math_helpers
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import robot_command_pb2, basic_command_pb2, image_pb2
from bosdyn.api import geometry_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pd2
from bosdyn.util import seconds_to_duration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from model_util_sunrgbd import SunrgbdDatasetConfig
from ap_helper import parse_predictions

parser = argparse.ArgumentParser()
bosdyn.client.util.add_base_arguments(parser)
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
# robot
parser.add_argument("--username", type=str, default="user", help="Username of Spot")
parser.add_argument("--password", type=str, default="97qp5bwpwf2c", help="Password of Spot")  # dungnydsc8su
parser.add_argument("--dock_id", type=int, default="521", help="Docking station ID to dock at")
parser.add_argument("--time_per_move", type=int, default=10, help="Seconds each move in grid should take")
parser.add_argument('--image-service', help='Name of the image service to query.',
                    default=ImageClient.default_service_name)
FLAGS = parser.parse_args()
bosdyn.client.util.setup_logging(FLAGS.verbose)

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
BBOX_RESULT = ["all", "confident", "nms", "confident_nms"][3]
FRONT_TRUNC = 0.1
DUMP_CONF_THRESH = 0.50  # Dump boxes with obj prob larger than that.
GROUND_PERCENTILE = 10
GROUND_BIAS = 0.03

FRONT_CAM_ANGLE = 15
BODY_TO_RIGHT_CAM = np.array([-0.14, -0.12, 0.13])

# robot image
ROTATION_ANGLE = {
    'hand_color_image': 0,
    'hand_depth': -90,
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180,
    'right_depth': 180
}


def get_depth(depth_img=None):
    # unit: mm
    if depth_img is None:
        # depth_img = "chairs/right_depth_black.png"
        depth_img = "robot_image/right_depth.png"
    depth = o3d.io.read_image(depth_img)
    return depth


def get_pcd(src="depth", to_np=True, remove_ground=False, depth_img=None):
    if src == "pcd":
        sample_id = 1
        depth_path = os.path.join("sunrgbd-toy", "sunrgbd_pc_bbox_votes_50k_v1_val/{:06d}_pc.npz".format(sample_id))
        points = np.load(depth_path)['pc'][:, :3]
        if to_np:
            return points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    depth = []
    if src == "depth":
        depth = get_depth(depth_img=depth_img)
    height, width, *_ = np.asarray(depth).shape
    # TODO
    fov_y = 60  # top to bottom
    fx = fy = 0.5 * height / math.tan(math.radians(fov_y * 0.5))
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth,
        intrinsic=intrinsic,
        extrinsic=np.eye(4).astype(np.float32),
        depth_scale=1000,
        depth_trunc=4,
    )
    # camera tilt
    pcd_pts = np.asarray(pcd.points)
    pcd_pts = pcd_pts[pcd_pts[:, -1] > FRONT_TRUNC]
    pcd.points = o3d.utility.Vector3dVector(pcd_pts)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # to sunrgbd, rotate along x
    rot_euler = np.array([math.radians(90 - FRONT_CAM_ANGLE), 0, 0])
    rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(rot_euler)
    pcd.rotate(rot_mat, (0, 0, 0))

    # remove ground
    if remove_ground:
        pcd_pts = np.asarray(pcd.points)
        ground_z = np.percentile(pcd_pts[:, -1], GROUND_PERCENTILE)
        pcd_pts = pcd_pts[pcd_pts[:, -1] > ground_z + GROUND_BIAS]
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)

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


def get_model_input(src="depth", to_np=True, remove_ground=False, depth_img=None):
    pcd = get_pcd(src=src, to_np=to_np, remove_ground=remove_ground, depth_img=depth_img)
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


def predict(net, pcd, dump=False):
    net.eval()  # set model to eval mode (for bn and dp)
    inputs = {'point_clouds': pcd}
    with torch.no_grad():
        end_points = net(inputs)
    end_points.update(inputs)
    parse_predictions(end_points, CONFIG_DICT, KEY_PREFIX_LIST[0])

    if not dump:
        return get_pred_bbox(end_points, DATASET_CONFIG, key_prefix=KEY_PREFIX_LIST[-1], already_numpy=False)

    MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG, inference_switch=True, key_prefix=KEY_PREFIX_LIST[-1])
    
    for k, v in end_points.items():
        if isinstance(v, torch.Tensor):
            end_points[k] = v.detach().cpu().numpy()
        else:
            end_points[k] = np.array(v, dtype=object)
    np.savez("pred.npz", **end_points)
    return parse_result()


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
    i = 0  # batch
    objectness_prob = softmax(objectness_scores[i, :, :])[:, 1]  # (K,)

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
            argidx = np.argsort(objectness_prob)[::-1]
            objectness_prob = objectness_prob[argidx]
            obbs = np.vstack(tuple(obbs))  # (num_proposal, 7)
            obbs = obbs[argidx]
            classes = np.array(classes)[argidx]

            selected = None
            if BBOX_RESULT == "all":
                selected = np.arange(obbs.shape[0])
            elif BBOX_RESULT == "confident":
                selected = objectness_prob > DUMP_CONF_THRESH
            elif BBOX_RESULT == "nms":
                selected = pred_mask[i, :] == 1
            elif BBOX_RESULT == "confident_nms":
                selected = np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[i, :] == 1)
            obbs = obbs[selected]
            objectness_prob = objectness_prob[selected]
            classes = list(map(lambda x: config.class2type[x], classes[selected]))
            print("centers: ", [x[:3] for x in obbs])
            print("radius: ", [x[3: 6] for x in obbs])
            print("confidence: ", objectness_prob)
            print("class: ", classes)

            return obbs, classes, objectness_prob
    print("no detection bbox")
    return [], [], []


def parse_result(result_path="pred.npz"):
    res = dict(np.load(result_path, allow_pickle=True))
    confident_nms_obbs, classes, objectness_prob = get_pred_bbox(res, DATASET_CONFIG, key_prefix=KEY_PREFIX_LIST[-1], already_numpy=True)
    return confident_nms_obbs, classes, objectness_prob


def get_3d_bbox(bboxes_3d, top_k=1):
    o3d_bboxes = []
    for bbox_3d in bboxes_3d[:top_k]:
        # https://github.com/isl-org/Open3D/issues/2
        # text viz
        # rotation is from x to -y
        rot_euler = np.array([0, 0, math.radians(-bbox_3d[6])])
        rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(rot_euler)
        o3d_bbox = o3d.geometry.OrientedBoundingBox(center=bbox_3d[:3], R=rot_mat, extent=2 * bbox_3d[3:6])
        o3d_bboxes.append(o3d_bbox)

    return o3d_bboxes


def viz_result(remove_ground=True, top_k=1):
    pcd = get_pcd(to_np=False, remove_ground=remove_ground)
    confident_nms_obbs, classes, objectness_prob = parse_result()
    bboxes = get_3d_bbox(confident_nms_obbs, top_k=top_k)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame, *bboxes], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)


def make_prediction(net=None, depth_img=None, dump=False):
    print("try to get point cloud")
    pcd = torch.tensor(get_model_input(depth_img=depth_img), dtype=torch.float32).to(device)
    print("got point cloud")
    print("try to get network")
    if net is None:
        net = get_model().to(device)
    print("got network")
    print("try to predict")
    return predict(net, pcd, dump=dump)


def crop_object(pcd, bbox, path="crop.ply"):
    print("crop ", path)
    pcd = o3d.geometry.PointCloud.crop(pcd, bbox)
    o3d.io.write_point_cloud(path, pcd, write_ascii=True, print_progress=True)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame, bbox], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)
    return pcd


def crop_result():
    pcd = get_pcd(to_np=False, remove_ground=True)
    confident_nms_obbs, classes, objectness_prob = parse_result()
    if not classes:
        print("no result")
        return
    cls = classes[0]
    bbox = get_3d_bbox(confident_nms_obbs)[0]
    pcd = crop_object(pcd, bbox, cls + ".ply")
    return pcd


def move_robot(robot, robot_state_client, robot_command_client, config,
               pos_vision, rot_vision, is_start=True, is_end=True, rotate_before_move=False):
    # Power on
    if not robot.is_powered_on():
        robot.logger.info("Powering on robot... This may take several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

    if is_start:
        # Undock
        try:
            robot.logger.info("Robot undocking...\nCLEAR AREA in front of docking station.")
            blocking_undock(robot)
            robot.logger.info("Robot undocked and standing")
            time.sleep(1)
        except Exception as e:
            pass

        # Stand
        robot.logger.info("Commanding robot to stand...")
        blocking_stand(robot_command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")
        time.sleep(3)

    # Initialize a robot command message, which we will build out below
    command = robot_command_pb2.RobotCommand()

    time_full = config.time_per_move
    if rotate_before_move:
        point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
        point.pose.position.x, point.pose.position.y = pos_vision[0], pos_vision[1]  # only x, y
        point.pose.angle = 0
        point.time_since_reference.CopyFrom(seconds_to_duration(time_full))
        time_full += config.time_per_move

    point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
    point.pose.position.x, point.pose.position.y = pos_vision[0], pos_vision[1]  # only x, y
    point.pose.angle = yaw_angle(rot_vision)
    point.time_since_reference.CopyFrom(seconds_to_duration(time_full))

    command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = VISION_FRAME_NAME
    robot.logger.info("Send body trajectory command.")
    cmd_id = robot_command_client.robot_command(command, end_time_secs=time.time() + time_full)
    time.sleep(time_full + 2)
    block_for_trajectory_cmd(command_client=robot_command_client, cmd_id=cmd_id,
                             body_movement_statuses={
                                 basic_command_pb2.SE2TrajectoryCommand.Feedback.BODY_STATUS_SETTLED},
                             timeout_sec=2,
                             logger=robot.logger)

    if is_end:
        # Dock robot after mission complete
        blocking_dock_robot(robot, config.dock_id)
        robot.logger.info("Robot docked")

        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed"
        robot.logger.info("Robot safely powered off")


def yaw_angle(rot_vision):
    if len(rot_vision) == 3:  # euler angel
        return math.radians(rot_vision[2])  # only yaw
    else:  # quaternion
        return math_helpers.quat_to_eulerZYX(math_helpers.Quat(*rot_vision))[0]  # only yaw


def init_robot(config):
    sdk = create_standard_sdk("move_robot_base")
    robot = sdk.create_robot(config.hostname)

    robot.authenticate(username=config.username, password=config.password)
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    return robot, robot_state_client, robot_command_client, lease_client


def init_image_capture(config):
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(config.hostname)
    robot.authenticate(username=config.username, password=config.password)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(config.image_service)
    return image_client


def init_state_client(config):
    sdk = bosdyn.client.create_standard_sdk('RobotStateClient')
    robot = sdk.create_robot(config.hostname)
    robot.authenticate(username=config.username, password=config.password)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    return robot_state_client


def get_state(robot_state_client):
    state = robot_state_client.get_robot_state()
    # arm_and_mobility_command.py
    # floor is z=-0.15
    # lowest stand z=0.14
    # highest stand z=0.36
    # x, y, z: center of bottom
    # safe region (gray mattress)
    # 2 <= x <= 5
    # -1.5 <= y <= 1.3
    vision_t_world = get_vision_tform_body(state.kinematic_state.transforms_snapshot)

    return vision_t_world


def sunrgbd2spot(pos):
    pos = deepcopy(pos)
    pos[0], pos[1] = pos[1], -pos[0]
    return pos


def spot2sunrgbd(pos):
    pos = deepcopy(pos)
    pos[0], pos[1] = -pos[1], pos[0]
    return pos


def extract_pos_rotation(state):
    body_center_pos = np.array([state.x, state.y, state.z])
    quaternion = np.array([state.rot.w, state.rot.x, state.rot.y, state.rot.z])
    rot = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)

    right_camera_pos = body_center_pos + rot @ BODY_TO_RIGHT_CAM
    return body_center_pos, rot, right_camera_pos


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


def capture_robot_image(image_client, pixel_fotmat="PIXEL_FORMAT_DEPTH_U16", image_source="right_depth",
                        image_saved_folder="robot_image", show_img=False):
    pixel_format = pixel_format_string_to_enum(pixel_fotmat)
    image_request = [
        build_image_request(image_source, pixel_format=pixel_format)
    ]
    image_responses = image_client.get_image(image_request)
    image = image_responses[0]
    num_bytes = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = ".png"
    else:
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_bytes = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_bytes = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_bytes = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_bytes = 2
        dtype = np.uint8
        extension = ".jpg"

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    # auto rotate
    img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

    if not os.path.exists(image_saved_folder):
        os.mkdir(image_saved_folder)
    image_saved_path = os.path.join(image_saved_folder, image_source + extension)
    cv2.imwrite(image_saved_path, img)
    if show_img:
        window_name = image_source
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    return img, image_saved_path


def detect_and_go(wait_for_result=True):
    net = get_model().to(device)
    robot, robot_state_client, robot_command_client, lease_client = init_robot(FLAGS)
    image_client = init_image_capture(FLAGS)
    robot_state_client = init_state_client(FLAGS)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # init pos
        robot.logger.info("Robot is starting")
        pos_vision, rot_vision = (3, 0, 0), (0, 0, 90)
        move_robot(robot, robot_state_client, robot_command_client, FLAGS,
                   pos_vision, rot_vision, is_start=True, is_end=False, rotate_before_move=True)
        # detect
        while True:
            state = get_state(robot_state_client)
            _, img_path = capture_robot_image(image_client, show_img=True)
            confident_nms_obbs, classes, objectness_prob = make_prediction(net=net, depth_img=img_path, dump=False)
            if len(classes) == 0:
                print("no detection")
                if wait_for_result:
                    continue
            else:
                center = confident_nms_obbs[0][:2]
                # assume camera to body center is 0.25m
                pos_vision, rot_vision = (3 + center[1] - 1, 0 - center[0] - 0.15, 0), (0, 0, 0)
                input("move to {}".format(pos_vision))
                move_robot(robot, robot_state_client, robot_command_client, FLAGS,
                           pos_vision, rot_vision, is_start=False, is_end=False, rotate_before_move=True)
            break

        # end
        robot.logger.info("Robot is going back")
        pos_vision, rot_vision = (2, 0, 0), (0, 0, 0)
        move_robot(robot, robot_state_client, robot_command_client, FLAGS,
                   pos_vision, rot_vision, is_start=False, is_end=True)


if __name__ == "__main__":
    # make_prediction(dump=True)
    # viz_result(top_k=2)
    # viz_full_pcd()
    # crop_result()
    # detect_and_go()
    pass
