import open3d as o3d
import numpy as np
import math


USE_HEIGHT = True


def get_depth():
    # unit: mm
    depth = o3d.io.read_image("sample-image/depth.png")
    return depth


def get_pcd():
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
    # pcd.rotate(rot_mat, (0, 0, 0))

    return pcd


def viz_pcd():
    pcd = get_pcd()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)


if __name__ == "__main__":
    viz_pcd()
    pass
