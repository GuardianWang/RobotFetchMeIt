import open3d as o3d
import numpy as np
import math


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
    return pcd


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
    DATASET_CONFIG = SunrgbdDatasetConfig()
    MODEL = importlib.import_module('votenet')
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


if __name__ == "__main__":
    viz_model_input()
    pass
