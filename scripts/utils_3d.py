import argparse
import os
from os.path import exists, join

import cv2
import gin
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm


@gin.configurable
def fuse_mesh(
        scan_dir: str,
        sdf_trunc: float = 0.06,
        voxel_length: float = 0.02,
        depth_trunc: float = 3.0,
        depth_scale: float = 1000.0,
):
    color_dir = join(scan_dir, 'color')
    depth_dir = join(scan_dir, 'depth')
    pose_dir = join(scan_dir, 'pose')
    intrinsic_dir = join(scan_dir, 'intrinsic')

    assert exists(color_dir)
    assert exists(depth_dir)
    assert exists(pose_dir)
    assert exists(intrinsic_dir)
    print("hello1")
    color_list = os.listdir(color_dir)
    color_list.sort(key=lambda e: int(e[:-4]))

    depth_list = os.listdir(depth_dir)
    depth_list.sort(key=lambda e: int(e[:-4]))

    pose_list = os.listdir(pose_dir)
    pose_list.sort(key=lambda e: int(e[:-4]))

    intr_list = os.listdir(intrinsic_dir)
    intr_list.sort(key=lambda e: int(e[:-4]))
    print("hello2")
    # see if all files exists
    assert all(
        (a[:-4] == b[:-4]) and (a[:-4] == c[:-4]) and (a[:-4] == d[:-4])
        for a, b, c, d in zip(color_list, depth_list, pose_list, intr_list))
    ### sdf_trunc 指定了这个截断范围的大小
    ### 有助于平滑表面，减少远距离或噪声点云对重建结果的影响
    ## voxel_length定义了用于构建TSDF体积网格的每个立方体“体素”（voxel）的边长
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        sdf_trunc=sdf_trunc,
        voxel_length=voxel_length,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    print("hello3")
    # i = 0
    for color_f, depth_f, pose_f, intr_f in tqdm(
            zip(color_list, depth_list, pose_list, intr_list),
            total=len(color_list),
    ):
        # if i > 10:
        #     break
        # i += 1
        intr = np.loadtxt(join(intrinsic_dir, intr_f))
        pose = np.loadtxt(join(pose_dir, pose_f))
        color = np.asanyarray(Image.open(join(color_dir, color_f))).astype(np.uint8)
        depth = np.asarray(Image.open(join(depth_dir, depth_f))).astype(np.uint16)

        h, w, _ = color.shape
        color = o3d.geometry.Image(color)
        depth = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color,
            depth=depth,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        print("hello4")
        # print(rgbd.shape)  # 应该是 (h, w, 4) 或类似格式
        # print(intr.shape)  # 应该是 (3, 3)
        # print(pose.shape)  # 应该是 (4, 4)
        # print(np.linalg.inv(pose).shape)  # 应该是 (4, 4)
        # print("intr[1, 2]:", intr[1, 2])
        # print("Intr values:", intr)  # 确保没有 NaN 或 Inf
        # print("Pose values:", pose)  # 确保没有 NaN 或 Inf
        # print("Extrinsic values:", np.linalg.inv(pose))  # 检查逆矩阵是否正确
        tsdf.integrate(
            image=rgbd,
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                height=h,
                width=w,
                fx=intr[0, 0],
                fy=intr[1, 1],
                cx=intr[0, 2],
                cy=intr[1, 2]
            ),
            extrinsic=np.linalg.inv(pose),  #
        )
        print("hello4.1")
    print("hello5")
    mesh = tsdf.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(join(scan_dir, 'mesh.ply'), mesh)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str)
    parser.add_argument("--sdf_trunc", type=float, default=0.04)
    parser.add_argument("--voxel_length", type=float, default=0.008)
    parser.add_argument("--depth_trunc", type=float, default=3.0)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument('--config', help='Name of config file')

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)
    fuse_mesh(
        scan_dir=args.workspace,
        sdf_trunc=args.sdf_trunc,
        voxel_length=args.voxel_length,
        depth_trunc=args.depth_trunc,
        depth_scale=args.depth_scale,
    )
