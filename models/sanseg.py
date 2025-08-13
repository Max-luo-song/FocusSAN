import os
import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Union

from os.path import abspath, dirname, join

import cv2
import gin
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultTrainer
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from PIL import Image
sys.path.append("3rdparty/SAN")
from labelmaker.label_data import get_ade150, get_replica, get_wordnet, get_occ11

# sys.path.append(
#     os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'SAN'))

from san import add_san_config

logging.basicConfig(level="INFO")
log = logging.getLogger('SAN Segmentation')


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def setup(config_file: str, device=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device or "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg

def load_san(
        device: Union[str, torch.device],
):
    """

    Args:
        device:
        custom_templates:

    Returns:
        Return san model
    """
    config_file = str(
        # abspath(
        #     join(__file__, '../..', '3rdparty', 'SAN', 'configs',
        #          'san_clip_vit_large_res4_coco.yaml')
        # )
        "/data3/gls/code/habitate_demo/SAN/configs/san_clip_vit_large_res4_coco.yaml"
    )

    model_path = abspath(
        # join(__file__, '../..', 'checkpoints', 'san_vit_large_14.pth')
        "/data3/gls/code/habitate_demo/SAN/model/san_vit_large_14.pth"
    )

    # cfg = setup(config_file)
    # model = DefaultTrainer.build_model(cfg)
    # if model_path.startswith("huggingface:"):
    #     model_path = download_model(model_path)
    # # print("Loading model from: ", model_path)
    # # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    # #     model_path
    # # )
    # # print("Loaded model from: ", model_path)
    # model.eval()
    # if torch.cuda.is_available():
    #     model = model.cuda()

    # # log.info('[SAN] Loading model from: {}'.format(model_path))
    # # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    # #     model_path
    # # )
    # model.eval()

    # cfg = setup(config_file)
    # model = DefaultTrainer.build_model(cfg)
    # if model_path.startswith("huggingface:"):
    #     model_path = download_model(model_path)
    # print("Loading model from: ", model_path)
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     model_path
    # )
    # print("Loaded model from: ", model_path)
    # model.eval()
    # if torch.cuda.is_available():
    #     model = model.cuda()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_san_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    model = DefaultTrainer.build_model(cfg)

    log.info('[SAN] Loading model from: {}'.format(model_path))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        model_path
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model

    return model


def process_image(
        model,
        img_path,
        class_names,
        threshold=0.7,
        flip=False,
):
    # preprocess
    img = Image.open(img_path).convert("RGB")
    if flip:
        img = Image.fromarray(np.array(img)[:, ::-1])
    w, h = img.size
    img = torch.from_numpy(np.asarray(img).copy()).float()
    temp = np.asarray(img)
    # img = img / 255.0
    img = img.permute(2, 0, 1) # 3 H W
    # 将PyTorch张量转换为NumPy数组
    img_np = img.numpy()

    # 将NumPy数组转换为字符串
    img_str = np.array2string(temp, separator=', ')
    # 写入到test.txt文件
    with open('/data3/gls/code/LabelMaker-main/render_20250715_around/1/test/test.txt', 'w') as f:
        f.write(str(img_np.shape))
        f.write(img_str)
    # model inference
    with torch.no_grad():
        predictions = model(
            [
                {
                    "image": img,
                    "height": h,
                    "width": w,
                    "vocabulary": class_names,
                }
            ]
        )[0]

    # torch.
    # torch.softmax(predictions['sem_seg'], dim=0)
    # product, pred = torch.max(torch.softmax(predictions['sem_seg'],dim=0), dim=0)
    product, pred = torch.max(predictions['sem_seg'], dim=0)


    # 0 is empty
    # ceiling 0+1 =1
    pred = pred + 1

    # pred[pred >= len(class_names)] = len(class_names)

    # map unknown region to 0
    pred[product < threshold] = 0

    pred = pred.to('cpu').numpy().astype(int)

    if flip:
        pred = pred[:, ::-1]

    # pred = pred + 1

    return pred


def get_vocabulary(classes):
    if classes == 'occ11':
        classes_data = get_occ11()

        # 按照 'id' 排序
        sorted_data = sorted(classes_data, key=lambda x: x['id'])

        # 提取 'name' 字段
        vocabulary = [item['name'] for item in sorted_data]
        # 提取color字段
        color = [item['color'] for item in sorted_data]
        # 移除empty
        vocabulary = vocabulary[1:]
        color = color[1:]

    else:
        raise ValueError(f'Unknown class set {classes}')
    return vocabulary, color


@gin.configurable
def run(
        scene_dir: Union[str, Path],
        output_folder: Union[str, Path],
        device: Union[str, torch.device] = 'cuda:0',
        # changing this to cuda default as all of us have it available. Otherwise, it will fail on machines without cuda
        classes='occ11',  # for open vocabulary method
        flip: bool = False,
):
    # convert str to Path object
    scene_dir = Path(scene_dir)
    output_folder = Path(output_folder)

    # check if scene_dir exists
    assert scene_dir.exists() and scene_dir.is_dir()

    input_color_dir = scene_dir / 'colors'
    assert input_color_dir.exists() and input_color_dir.is_dir()

    output_dir = scene_dir / output_folder
    output_dir = Path(str(output_dir) + '_flip') if flip else output_dir

    # only for open vocabulary method
    if classes != 'occ11':
        output_dir.replace('occ11', classes)

    # check if output directory exists
    shutil.rmtree(output_dir, ignore_errors=True)  # remove output_dir if it exists
    os.makedirs(str(output_dir), exist_ok=False)

    input_files = input_color_dir.glob('*')
    input_files = sorted(input_files, key=lambda x: int(x.stem.split('_')[-1]))

    log.info(f'[san] using {classes} classes')
    log.info(f'[san] inference in {str(input_color_dir)}')

    # templates, class_names = get_templates(classes)
    # id_map = get_id_map(classes)

    class_names, colors = get_vocabulary(classes)

    log.info('[san] loading model')
    model = load_san(device=device)

    log.info('[san] inference')

    for file in tqdm(input_files):
        result = process_image(model, file, class_names, flip=flip)
        print("class_names:", class_names)
        print("color:", colors)
        # print("result:", result)
    #     cv2.imwrite(
    #         str(output_dir / f'{file.stem}.png'),
    #         result.astype(np.uint8),
    #     )
        color_result = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        
        # 遍历每个类别编号，将对应的颜色填充到color_result中
        for idx, color in enumerate(colors):
            # 找到当前类别的所有像素位置
            mask = (result == idx)
            # 将这些位置的颜色设置为预定义的颜色
            color_result[mask] = color
        
        # 保存颜色编码的语义图
        cv2.imwrite(
            str(output_dir / f'{file.stem}.png'),
            color_result
        )


def arg_parser():
    parser = argparse.ArgumentParser(description='SAN Segmentation')
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
        required=True,
        help=
        'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version',
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--flip',
        action="store_true",
        help='Flip the input image, this is part of test time augmentation.',
    )
    parser.add_argument('--config', help='Name of config file')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)

    setup_seeds(seed=args.seed)
    # workspace参数，把原始rgb数据放到workspace下color文件夹
    run(scene_dir=args.workspace, output_folder=args.output, flip=args.flip)
