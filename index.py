# -*- coding: utf-8 -*-

import h5py
import argparse
import numpy as np
from service.vggnet import VGGNet
import os
import sys
from os.path import dirname
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QFileDialog

BASE_DIR = dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def get_imlist(path):
    """
    获得文件夹中所有图像的文件路径
    Args:
        path: 图像所在文件夹路径
    Returns:图像文件路径列表
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QWidget()
    train_data_dir = QFileDialog.getExistingDirectory(widget, '请选择训练图像所在文件夹！', 'home')
    print('训练图像文件夹路径为：{}'.format(train_data_dir))

    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_data", type=str, default=os.path.join(BASE_DIR, 'data', 'image_mtaching_5000'),
    #                     help="train data path.")  # 定义训练集所在路径
    parser.add_argument("--train_data", type=str, default=train_data_dir,
                        help="train data path.")  # 定义训练集所在路径
    parser.add_argument("--index_file", type=str, default=os.path.join(BASE_DIR, 'index', 'train.h5'),
                        help="index file path.")
    args = vars(parser.parse_args())
    img_list = get_imlist(args["train_data"])
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    feats = []
    names = []
    model = VGGNet()  # 定义特征提取模型为VGGNet模型
    for i, img_path in enumerate(img_list):
        norm_feat = model.vgg_extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))
    feats = np.array(feats)
    print("--------------------------------------------------")
    print("         writing feature extraction results")
    print("--------------------------------------------------")
    h5f = h5py.File(args["index_file"], 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(names))

    h5f.close()

    sys.exit(app.exec_())
