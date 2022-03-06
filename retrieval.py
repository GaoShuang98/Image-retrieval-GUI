# -*- coding: utf-8 -*-

import argparse
from service.vggnet import VGGNet
from service.numpy_retrieval import NumpyRetrieval
from service.faiss_retrieval import FaissRetrieval
from service.es_retrieval import ESRetrieval
from service.milvus_retrieval import MilvusRetrieval
import os
import sys
from os.path import dirname
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QFileDialog

BASE_DIR = dirname(os.path.abspath(__file__))  # 获得.py文件的相对路径
sys.path.append(BASE_DIR)  # 添加引用模块的地址（运行后地址失效）


class RetrievalEngine(object):

    def __init__(self, index_file, db_name):
        self.index_file = index_file
        self.db_name = db_name
        self.numpy_r = self.faiss_r = self.es_r = self.milvus_r = None

    def get_method(self, m_name):
        m_name = "%s_handler" % str(m_name)
        method = getattr(self, m_name, self.default_handler)
        return method

    def numpy_handler(self, query_vector, req_id=None):
        # numpy计算
        if self.numpy_r is None:
            self.numpy_r = NumpyRetrieval(self.index_file)
        return self.numpy_r.retrieve(query_vector)

    def faiss_handler(self, query_vector, req_id=None):
        # faiss计算
        if self.faiss_r is None:
            self.faiss_r = FaissRetrieval(self.index_file)  # 类的实例化
        return self.faiss_r.retrieve(query_vector)  # 调用faiss类中的方法

    def es_handler(self, query_vector, req_id=None):
        # es计算
        if self.es_r is None:
            self.es_r = ESRetrieval(self.db_name, self.index_file)
        return self.es_r.retrieve(query_vector)

    def milvus_handler(self, query_vector, req_id=None):
        # milvus计算
        if self.milvus_r is None:
            self.milvus_r = MilvusRetrieval(self.db_name, self.index_file)
        return self.milvus_r.retrieve(query_vector)

    def default_handler(self, query_vector, req_id=None):
        return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--test_data", type=str, default=os.path.join(BASE_DIR, 'data', 'test', '001_accordion_image_0001.jpg'), help="test data path.")

    app = QApplication(sys.argv)
    widget = QWidget()
    retrieval_data_dir = QFileDialog.getOpenFileName(widget, '请选择待查找图像所在文件夹！', 'home', filter='*.jpg *.png')[0]
    print('待查找图像文件夹路径为：{}'.format(retrieval_data_dir))
    parser.add_argument("--test_data", type=str,
                        default=retrieval_data_dir,
                        help="test data path.")
    parser.add_argument("--index_file", type=str, default=os.path.join(BASE_DIR, 'index', 'train.h5'),
                        help="index file path.")
    parser.add_argument("--db_name", type=str, default='image-retrieval', help="database name.")
    # parser.add_argument("--engine", type=str, default='numpy', help="retrieval engine.")
    parser.add_argument("--engine", type=str, default='faiss', help="retrieval engine.")
    args = vars(parser.parse_args())
    # 1.图片推理
    model = VGGNet()
    query_vector = model.vgg_extract_feat(args["test_data"])  # 利用model.vgg_extract_feat()函数生成待查询图像的特征向量
    # 2.图片检索
    re = RetrievalEngine(args["index_file"], args["db_name"])  # 传入参数 index_file, db_name
    result = re.get_method(args["engine"])(query_vector, None)

    parser.add_argument('--retrieval_pic_1', type=str, default=os.path.join(BASE_DIR, ))
    print(result)

    sys.exit(app.exec())
