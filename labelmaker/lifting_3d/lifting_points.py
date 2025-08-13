import argparse
import logging
import os
from pathlib import Path
from typing import Union

import cv2
import gin
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append("/data3/gls/code/LabelMaker-main")

from labelmaker.label_data import get_wordnet, get_occ11

logging.basicConfig(level="INFO")
log = logging.getLogger('3D Point Lifting')


Scanetv2_40_colors_rgb: np.ndarray = np.array(
    [
        [255, 255, 0],        # 0
        [174, 199, 232],  # 1
        [152, 223, 138],  # 2
        [31, 119, 180],   # 3
        [255, 187, 120],  # 4
        [188, 189, 34],   # 5
        [91, 135, 229],   # 6
        [255, 152, 150],  # 7
        [214, 39, 40],    # 8
        [197, 176, 213],  # 9
        [148, 103, 189], # 10
        [196, 156, 148], # 11
        [56, 144, 89],   # 12
        [123, 172, 47],  # 13
        [247, 182, 210], # 14
        [229, 91, 104],  # 15
        [219, 219, 141], # 16
        [16, 212, 139],  # 17
        [141, 91, 229],  # 18
        [162, 62, 60],   # 19
        [184, 58, 125],  # 20
        [34, 14, 130],   # 21
        [186, 197, 62],  # 22
        [192, 229, 91],  # 23
        [255, 127, 14],  # 24
        [143, 45, 115],  # 25
        [25, 160, 151],  # 26
        [62, 143, 148],  # 27
        [158, 218, 229], # 28
        [177, 82, 239],  # 29
        [237, 80, 38],   # 30
        [82, 75, 227],   # 31
        [137, 63, 14],   # 32
        [44, 160, 44],   # 33
        [112, 128, 144], # 34
        [237, 204, 37],  # 35
        [227, 119, 194], # 36
        [169, 137, 78],  # 37
        [20, 65, 247],   # 38
        [38, 96, 167],   # 39
        [98, 141, 208]   # 40
    ],
    dtype=np.uint8,
)

from scipy.spatial import KDTree
# 构建 KDTree 用于快速查找最近的 RGB 值
color_tree = KDTree(Scanetv2_40_colors_rgb)

def rgb_to_label(rgb_array):
    """
    将 RGB 数组转换为类别编号
    :param rgb_array: 形状为 (H, W, 3) 的 RGB 数组
    :return: 形状为 (H, W) 的类别编号数组
    """
    # 将 RGB 数组展平为 (H*W, 3)
    flat_rgb = rgb_array.reshape(-1, 3)
    # 查找每个 RGB 值最近的类别编号
    _, indices = color_tree.query(flat_rgb)
    # 将结果 reshape 回 (H, W)
    return indices.reshape(rgb_array.shape[:2])

def project_pointcloud(points, pose, intrinsics):
    """将3D点云投影到2D图像平面

    该函数通过给定的相机位姿和相机内参，将3D点云投影到2D图像平面。
    计算过程包括：将点云转换到相机坐标系，然后应用相机内参进行投影，
    最后进行归一化处理得到2D坐标。

    Args:
        points: numpy数组，形状为(N,3)，表示N个3D点的坐标(x,y,z)
        pose: numpy数组，形状为(4,4)，表示相机在世界坐标系中的位姿(变换矩阵)
        intrinsics: numpy数组，形状为(3,3)或(3,4)，表示相机内参矩阵

    Returns:
        numpy数组，形状为(N,3)，表示投影后的2D齐次坐标(x,y,w)，
        其中w是深度值，x/w和y/w是实际的2D坐标
    """
    # 将3D点转换为齐次坐标(N,4)
    points_h = np.hstack((points, np.ones_like(points[:, 0:1])))
    # 将点从世界坐标系转换到相机坐标系
    points_c = np.linalg.inv(pose) @ points_h.T
    points_c = points_c.T
    # 如果内参矩阵是3x3，则扩展为4x4的齐次形式
    if intrinsics.shape[-1] == 3:
        intrinsics = np.hstack((intrinsics, np.zeros((3, 1))))
        intrinsics = np.vstack((intrinsics, np.zeros((1, 4))))
        intrinsics[-1, -1] = 1.
    # 应用内参矩阵进行投影变换
    points_p = intrinsics @ points_c.T
    points_p = points_p.T[:, :3]
    # 归一化处理，将齐次坐标转换为2D坐标
    points_p[:, 0] /= (points_p[:, -1] + 1.e-6)
    points_p[:, 1] /= (points_p[:, -1] + 1.e-6)

    return points_p


@gin.configurable
def main(
        scene_dir: Union[str, Path],
        label_folder: Union[str, Path],
        output_file: Union[str, Path],
        output_mesh: Union[str, Path],
        maximum_label: int,
        label_space='occ11',  # 原论文是wordnet

):
    scene_dir = Path(scene_dir)
    label_folder = Path(label_folder)
    output_file = Path(output_file)
    output_mesh = Path(output_mesh)

    # check if scene_dir exists
    assert scene_dir.exists() and scene_dir.is_dir()

    # define all paths
    input_color_dir = scene_dir / 'color'
    assert input_color_dir.exists() and input_color_dir.is_dir()

    input_depth_dir = scene_dir / 'depth'
    assert input_depth_dir.exists() and input_depth_dir.is_dir()

    input_intrinsic_dir = scene_dir / 'intrinsic'
    assert input_intrinsic_dir.exists() and input_intrinsic_dir.is_dir()

    input_pose_dir = scene_dir / 'pose'
    assert input_pose_dir.exists() and input_pose_dir.is_dir()

    input_label_dir = scene_dir / label_folder
    assert input_label_dir.exists() and input_label_dir.is_dir()

    input_mesh_path = scene_dir / 'mesh.ply'
    assert input_mesh_path.exists() and input_mesh_path.is_file()

    log.info('Processing {} using for labels {}'.format(
        str(scene_dir),
        str(input_label_dir),
    ))

    # load mesh and extract colors
    mesh = o3d.io.read_triangle_mesh(str(input_mesh_path))
    vertices = np.asarray(mesh.vertices)
    print("vertices shape:", vertices.shape)  # (1745135, 3)
    # init label container
    labels_3d = np.zeros((vertices.shape[0], maximum_label + 1))  
    print("labels_3d shape:", labels_3d.shape)
    files = input_label_dir.glob('*.png')
    files = sorted(files, key=lambda x: int(x.stem.split('.')[0]))
    resize_image = False

    # save colored mesh
    color_map = np.zeros(shape=(maximum_label, 3), dtype=np.uint8)

    if label_space == 'wordnet':
        for item in get_wordnet():
            color_map[item['id']] = item['color']
    elif label_space == 'occ11':
        for item in get_occ11():
            # print("max id:", np.max(item['id']))
            color_map[item['id']] = item['color']
    else:
        raise Exception(f'Unknown label space {label_space}')
    color_map = Scanetv2_40_colors_rgb
    # print("color_map:", color_map)
    print("color map shape:", color_map.shape)
    #### 构建自己的color_map

    # 把 3d mesh 再投影至 2D image, 再统计所有的
    for idx, file in tqdm(enumerate(files), total=len(files)):

        frame_key = file.stem
        frame_key = f'{int(frame_key) + 1:06d}'
        # 加载相机内参
        intrinsics = np.loadtxt(str(input_intrinsic_dir / f'{frame_key}.txt'))

        # 加载RGB图像
        image = np.asarray(Image.open(str(input_color_dir /
                                          f'{frame_key}.jpg'))).astype(np.uint8)  #
        # 加载深度图
        depth = np.asarray(Image.open(str(
            input_depth_dir / f'{frame_key}.png'))).astype(np.float32) / 1000.

        # 加载语义图
        labels = np.asarray(Image.open(str(file)))
        ### 把语义图根据RGB颜色转成(w, h, 1)
        rgb_labels = labels[..., :3]  # 提取前三个通道 (480, 640, 3)
        label_map = rgb_to_label(rgb_labels)  # 转换为 (480, 640)
        labels = label_map
        print("labels shape:", labels.shape)
        #
        max_label = np.max(labels)
        print("max_label:", max_label)
        if max_label > labels_3d.shape[-1] - 1:
            raise ValueError(
                f'Label {max_label} is not in the label range of {labels_3d.shape[-1]}'
            )

        # 图像缩放
        if resize_image:
            h, w = depth.shape
            image = cv2.resize(image, (w, h))
            labels = cv2.resize(labels, (w, h))
        else:
            h, w, _ = image.shape
            depth = cv2.resize(depth, (w, h))
        ### labels第三维应该是1

        print("labels shape:", labels.shape)
        # 加载相机位姿
        pose_file = input_pose_dir / f'{frame_key}.txt'
        pose = np.loadtxt(str(pose_file))

        points_p = project_pointcloud(vertices, pose, intrinsics)

        xx = points_p[:, 0].astype(int)
        yy = points_p[:, 1].astype(int)
        zz = points_p[:, 2]

        valid_mask = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)

        d = depth[yy[valid_mask], xx[valid_mask]]

        valid_mask[valid_mask] = (zz[valid_mask] > 0) & (np.abs(zz[valid_mask] - d) <= 0.1)
        ### labels_2d应该是rgb的编号
        labels_2d = labels[yy[valid_mask], xx[valid_mask]]
        
        print("valid_mask shape:", valid_mask.shape)
        print("labels_2d shape:", labels_2d.shape)
        print("labels_3d shape:", labels_3d.shape)
        labels_3d[valid_mask, labels_2d] += 1

    # extract labels
    labels_3d = np.argmax(labels_3d, axis=-1)
    print("save text path:", str(scene_dir / output_file))
    # save output
    np.savetxt(str(scene_dir / output_file), labels_3d, fmt='%i')  # 保存 3d label

    label_mesh_color = color_map[labels_3d]  # 根据color_map的映射来生成 mesh color

    label_mesh = o3d.geometry.TriangleMesh()
    label_mesh.vertices = mesh.vertices
    label_mesh.triangles = mesh.triangles

    label_mesh.vertex_colors = o3d.utility.Vector3dVector(
        label_mesh_color.astype(float) / 255)

    o3d.io.write_triangle_mesh(str(scene_dir / output_mesh), label_mesh)


def arg_parser():
    parser = argparse.ArgumentParser(
        description=
        'Project 3D points to 2D image plane and aggregate labels and save label txt'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        required=True,
        help=
        'Path to workspace directory. There should be a "color" folder inside.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='labels.txt',
        help='Name of files to save the labels',
    )
    parser.add_argument(
        '--output_mesh',
        type=str,
        default='point_lifted_mesh.ply',
        help='Name of files to save the labels',
    )
    parser.add_argument(
        '--label_folder',
        default='intermediate/consensus'
    )
    parser.add_argument(
        '--max_label',
        type=int,
        default=2000,
        help='Max label value',
    )
    parser.add_argument(
        '--config',
        help='Name of config file'
    )
    parser.add_argument(
        '--label_space',
        default='occ11'
    )  # ['wordnet','occ11']

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)
    main(
        scene_dir=args.workspace,
        label_folder=args.label_folder,
        output_file=args.output,
        output_mesh=args.output_mesh,
        maximum_label=args.max_label,
        label_space=args.label_space,
    )
