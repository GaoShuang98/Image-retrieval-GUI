# -*- coding: utf-8 -*-
import argparse
import math
import os
import sys
import time
from os.path import dirname

import h5py
import numpy
import numpy as np
import qdarkstyle
from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *

from image_retrieval_widget import Ui_MainWindow
from service.es_retrieval import ESRetrieval  # Elasticsearch
from service.faiss_retrieval import FaissRetrieval
from service.milvus_retrieval import MilvusRetrieval
from service.numpy_retrieval import NumpyRetrieval
from service.vggnet import VGGNet
from widget_progress_bar import Ui_widget_progress_bar


class RetrievalResult:
    def __init__(self):
        self.name = None
        self.full_name = None
        self.score = None


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


class IndexWorkThread(QThread):
    """利用多线程对图像特征进行提取，避免窗口卡死"""
    trigger = pyqtSignal(int)  # 自定义信号对象。参数str表示这个信号可以传入字符串

    def __init__(self, img_list, retrieval_DB_dir):
        super(IndexWorkThread, self).__init__()
        self.img_list = img_list
        self.retrieval_DB_dir = retrieval_DB_dir
        self.feats = []
        self.names = []

    def run(self):
        print('start_index(self) clicked!')
        print("--------------------------------------------------")
        print("         feature extraction starts")
        print("--------------------------------------------------")
        model = VGGNet()  # 定义特征提取模型为VGGNet模型
        total_img_num = len(self.img_list)
        for i, img_path in enumerate(self.img_list):
            norm_feat = model.vgg_extract_feat(img_path)
            img_name = os.path.split(img_path)[1]
            self.feats.append(norm_feat)
            self.names.append(img_name)
            self.trigger.emit(i + 1)
            print("extracting feature from image No. %d , %d images in total" % ((i + 1), total_img_num))
        self.save_h5py()
        self.trigger.emit(0)

    def save_h5py(self):
        self.feats = np.array(self.feats)
        print("--------------------------------------------------")
        print("         writing feature extraction results")
        print("--------------------------------------------------")

        h5f = h5py.File(self.retrieval_DB_dir, 'w')
        h5f.create_dataset('dataset_1', data=self.feats)
        h5f.create_dataset('dataset_2', data=np.string_(self.names))
        h5f.close()
        print('feature extraction results writing succeed')


class RetrievalProgram(QMainWindow, Ui_MainWindow, Ui_widget_progress_bar):  # 继承QMainWindow类，Ui_MainWindow类的属性
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        super().__init__()  # 继承父类必须使用的语句
        self.setupUi(self)  # 刷新self UI
        self.BASE_DIR = dirname(os.path.abspath(__file__))  # 获得.py文件的相对路径
        sys.path.append(self.BASE_DIR)

        self.pushButton_previous_pic.setEnabled(False)
        self.pushButton_next_pic.setEnabled(False)

        self.train_data_dir = r'E:\StreetData\UCF-Google-Streetview-II-Data\Google_Street-View_Images'  # 训练图像所在文件夹路径
        self.retrieval_data_dir = r"E:\StreetData\UCF-Google-Streetview-II-Data\Queries\1.jpg"  # 查询图像路径
        # self.retrieval_data_dir = r'E:\StreetData\UCF-Google-Streetview-II-Data\souls'
        self.parser = argparse.ArgumentParser()
        self.pause_signal = None  # 特征提取过程停止的信号变量
        self.retrieval_DB_dir = r"E:\StreetData\UCF-Google-Streetview-II-Data\image-retrieval\index\train(265490pics).h5"
        self.lineEdit_retrieval_DB_dir.setText(self.retrieval_DB_dir)  # 默认的搜索库路径写到lineEdit中
        self.db_name = 'index_and_retrieval_exe'
        self.retrieval_data_list = []  # 待搜索图像所在文件夹所有的图像路径list
        self.retrieval_data_index = None  # 当前待搜索图像在该文件夹list中的索引
        self.retrieval_data_dad_dir = None

    def get_retrieval_DB_dir(self):
        """
        变换默认的搜索库
        Returns:
        """
        self.retrieval_DB_dir = QFileDialog.getOpenFileName(self, caption='请选择搜索图像特征库', filter='*.h5')[0]
        print("self.retrieval_DB_dir {}".format(self.retrieval_DB_dir))
        self.lineEdit_retrieval_DB_dir.setText(self.retrieval_DB_dir)

    def get_imlist(self, path):
        """
        获得文件夹中所有图像的文件路径
        Args:
            path: 图像所在文件夹路径
        Returns:图像文件路径列表
        """
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

    def pic_double_clicked(self):
        print('pic_double_clicked(self)!')

    def get_img_dir(self):
        """
        获得训练图像文件并可视化到窗体上面
        Returns:

        """
        print('get_img_dir() clicked!')
        self.train_data_dir = os.path.dirname(
            QFileDialog.getOpenFileName(self, '请选择训练图像所在文件夹！', 'home', filter='*.jpg *.JPG *.png')[0])
        print('训练图像文件夹路径为：{}'.format(self.train_data_dir))
        if not self.train_data_dir:
            return

        self.img_list = self.get_imlist(self.train_data_dir)
        self.tableWidget.clearContents()  # 清空之前的tableWidget中所有内容
        total_img_num = len(self.img_list)
        # 对显示像片的tableWidget的一些属性进行设置
        self.tableWidget.setRowCount(math.ceil(total_img_num / 6))  # 设置tableWidget的总行数
        self.tableWidget.cellDoubleClicked['int', 'int'].connect(
            self.pic_double_clicked)  # 双击像片缩略图触发函数
        self.tableWidget.setShowGrid(False)  # 不显示网格
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 自适应列宽，非常重要
        self.tableWidget.setRowCount(math.ceil(total_img_num / 6))  # 除以6,有余数则加1，如math.ceil(14/6)=2
        self.tableWidget.setColumnCount(6)
        self.tableWidget.cellDoubleClicked['int', 'int'].connect(
            self.pic_double_clicked)  # 双击像片缩略图触发函数

        for i in range(6):  # 设置tableWidget的列宽
            self.tableWidget.setColumnWidth(i, 300)
        for i in range(math.ceil(total_img_num / 6)):  # 设置tableWidget的行高
            self.tableWidget.setRowHeight(i, 300)

        for i in range(total_img_num):  # 对每个像片进行遍历，显示到tableWidget中
            child1 = QTreeWidgetItem()
            child1.setText(0, self.img_list[i])
            # 建立label小单元用来显示像片
            img_pixmap = QPixmap(os.path.join(self.train_data_dir, self.img_list[i]))
            label = QLabel()  # 实例化label对象，用来放置像片
            label.setFixedSize(300, 250)  # 设置label的大小
            label.setStyleSheet("border:10px solid gray")
            label.setPixmap(img_pixmap)
            label.setScaledContents(True)
            # 实例化像片名称layout
            label_txt = QLabel()
            label_txt.setText(self.img_list[i].split('\\')[-1])  # label中设置像片名称
            Vlayout = QVBoxLayout()  # 实例化垂直layout，放置像片label与像片名称label
            Vlayout.addWidget(label, 0, Qt.AlignCenter)
            Vlayout.addWidget(label_txt, 0, Qt.AlignCenter)
            frame = QFrame()
            frame.setLayout(Vlayout)
            widget = QWidget()  # 实例化frame框架对象，用来放置像片label与像片名称label组成的layout
            widget.setLayout(Vlayout)  # 放置像片label与像片名称label组成的layout
            self.tableWidget.setCellWidget(i // 6, i % 6, widget)  # 将单个像片cellWidget显示到表格widget中

    def start_index(self):
        # 创建一个进度条（不能放的太靠前，应该放到循环附近，否则就会出现一段时间的：弹出来弹窗但是并没有进度条的状态）
        self.widget_progress_bar = Ui_widget_progress_bar()  # 实例化一个进度条窗口对象
        self.widget_progress_bar.setWindowTitle('图像特征向量提取中……')  # 修改进度条窗口的标题
        self.widget_progress_bar.progressBar.setMaximum(len(self.img_list))
        self.widget_progress_bar.progressBar.setValue(0)
        self.widget_progress_bar.show()
        QApplication.processEvents()  # 实时显示，非常重要！！！！！！！

        self.retrieval_DB_dir = os.path.join(self.BASE_DIR, 'index', 'train{}.h5'.format(str(time.strftime(
            '%Y-%m-%d_%H-%M-%S',
            time.localtime()))))  # 创建新的h5图像特征数据库文件
        self.lineEdit_retrieval_DB_dir.setText(self.retrieval_DB_dir)

        self.work = IndexWorkThread(self.img_list, self.retrieval_DB_dir)
        self.work.start()  # 启动线程
        self.work.trigger.connect(self.update_label)
        self.work.exit()

    def update_label(self, signal):
        if signal:
            self.widget_progress_bar.progressBar.setValue(signal)
            self.widget_progress_bar.feature_extract_info_label.setText("processing img No.{} ".format(str(signal)))
        else:
            self.widget_progress_bar.close()

    def pause_and_save(self):
        print('pause_and_save(self) clicked!')
        self.work.Pause = True
        self.work.save_h5py()
        self.work.exit()
        self.widget_progress_bar.close()

    def get_retrieval_img(self):
        print('get_retrieval_img(self) clicked!')
        self.retrieval_data_dir = QFileDialog.getOpenFileName(self, '请选择待搜索图像！', 'home', filter='*.jpg *.png')[0]
        print('待搜索图像文件路径为：{}'.format(self.retrieval_data_dir))
        if not self.retrieval_data_dir:
            return
        self.show_img_in_graphicview(self.graphicsView_retrieval_img, self.retrieval_data_dir)  # 显示待搜索图像
        self.label_retrieval_img.setText('待搜索图像：{}'.format(self.retrieval_data_dir))  # 修改待搜索图像下方label
        self.retrieval_data_dad_dir = os.path.dirname(self.retrieval_data_dir)  # 获得图像路径的父路径
        self.retrieval_data_list = os.listdir(self.retrieval_data_dad_dir)  # 获取图像所在文件夹中所有图像路径list
        self.retrieval_data_index = self.retrieval_data_list.index(
            os.path.basename(self.retrieval_data_dir))  # 获得当前待搜索图像在该文件夹list中的索引

        self.pushButton_previous_pic.setEnabled(True)
        self.pushButton_next_pic.setEnabled(True)

    def previous_pic(self):
        # print('pushbutton previous_pic clicked!')

        self.retrieval_data_index -= 1
        if self.retrieval_data_index == -1:
            self.retrieval_data_index = len(self.retrieval_data_list) - 1
        self.retrieval_DB_dir = self.lineEdit_retrieval_DB_dir.text()
        self.retrieval_data_dir = os.path.join(self.retrieval_data_dad_dir,
                                               self.retrieval_data_list[self.retrieval_data_index])
        self.show_img_in_graphicview(self.graphicsView_retrieval_img, self.retrieval_data_dir)  # 显示待搜索图像
        self.label_retrieval_img.setText('待搜索图像：{}'.format(self.retrieval_data_dir))  # 修改待搜索图像下方label
        # 1.图片推理
        model = VGGNet()
        query_vector = model.vgg_extract_feat(self.retrieval_data_dir)  # 利用model.vgg_extract_feat()函数生成待查询图像的特征向量
        # 2.图片检索
        re = RetrievalEngine(self.retrieval_DB_dir, self.db_name)  # 传入参数 index_file, db_name
        print(self.comboBox.currentText())
        result = re.get_method(self.comboBox.currentText())(query_vector)  # 根据所选择的图像检索方式进行检索

        pic_1 = RetrievalResult()
        pic_2 = RetrievalResult()
        pic_3 = RetrievalResult()

        self.graphicsView_retrieved_img_1.clearMask()
        self.graphicsView_retrieved_img_2.clearMask()
        self.graphicsView_retrieved_img_3.clearMask()
        try:
            pic_1.name = result[0]['name'].decode('ascii')
            pic_1.full_name = os.path.join(self.train_data_dir, pic_1.name)
            pic_1.score = result[0]['score']
            self.label_retrieved_img_1.setText(str(pic_1.score) + '|' + pic_1.name)
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_1, pic_1.full_name)
        except:
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_1, 'cross.png')
            self.label_retrieved_img_1.setText('无结果')
        try:
            pic_2.name = result[1]['name'].decode('ascii')
            pic_2.full_name = os.path.join(self.train_data_dir, pic_2.name)
            pic_2.score = result[1]['score']
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_2, pic_2.full_name)
            self.label_retrieved_img_2.setText(str(pic_2.score) + '|' + pic_2.name)
        except:
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_2, 'cross.png')
            self.label_retrieved_img_2.setText('无结果')
        try:
            pic_3.name = result[2]['name'].decode('ascii')
            pic_3.full_name = os.path.join(self.train_data_dir, pic_3.name)
            pic_3.score = result[2]['score']
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_3, pic_3.full_name)
            self.label_retrieved_img_3.setText(str(pic_3.score) + '|' + pic_3.name)
        except:
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_3, 'cross.png')
            self.label_retrieved_img_3.setText('无结果')

    def next_pic(self):
        # print('pushbutton next_pic clicked!')
        self.retrieval_data_index += 1
        if self.retrieval_data_index == len(self.retrieval_data_list):
            self.retrieval_data_index = 0
        self.retrieval_DB_dir = self.lineEdit_retrieval_DB_dir.text()
        self.retrieval_data_dir = os.path.join(self.retrieval_data_dad_dir,
                                               self.retrieval_data_list[self.retrieval_data_index])
        self.show_img_in_graphicview(self.graphicsView_retrieval_img, self.retrieval_data_dir)  # 显示待搜索图像
        self.label_retrieval_img.setText('待搜索图像：{}'.format(self.retrieval_data_dir))  # 修改待搜索图像下方label
        # 1.图片推理
        model = VGGNet()
        query_vector = model.vgg_extract_feat(self.retrieval_data_dir)  # 利用model.vgg_extract_feat()函数生成待查询图像的特征向量
        # 2.图片检索
        re = RetrievalEngine(self.retrieval_DB_dir, self.db_name)  # 传入参数 index_file, db_name
        print(self.comboBox.currentText())
        result = re.get_method(self.comboBox.currentText())(query_vector)  # 根据所选择的图像检索方式进行检索

        pic_1 = RetrievalResult()
        pic_2 = RetrievalResult()
        pic_3 = RetrievalResult()

        self.graphicsView_retrieved_img_1.clearMask()
        self.graphicsView_retrieved_img_2.clearMask()
        self.graphicsView_retrieved_img_3.clearMask()
        try:
            pic_1.name = result[0]['name'].decode('ascii')
            pic_1.full_name = os.path.join(self.train_data_dir, pic_1.name)
            pic_1.score = result[0]['score']
            self.label_retrieved_img_1.setText(str(pic_1.score) + '|' + pic_1.name)
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_1, pic_1.full_name)
        except:
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_1, 'cross.png')
            self.label_retrieved_img_1.setText('无结果')
        try:
            pic_2.name = result[1]['name'].decode('ascii')
            pic_2.full_name = os.path.join(self.train_data_dir, pic_2.name)
            pic_2.score = result[1]['score']
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_2, pic_2.full_name)
            self.label_retrieved_img_2.setText(str(pic_2.score) + '|' + pic_2.name)
        except:
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_2, 'cross.png')
            self.label_retrieved_img_2.setText('无结果')
        try:
            pic_3.name = result[2]['name'].decode('ascii')
            pic_3.full_name = os.path.join(self.train_data_dir, pic_3.name)
            pic_3.score = result[2]['score']
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_3, pic_3.full_name)
            self.label_retrieved_img_3.setText(str(pic_3.score) + '|' + pic_3.name)
        except:
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_3, 'cross.png')
            self.label_retrieved_img_3.setText('无结果')

    def show_img_in_graphicview(self, graphics_view, img_path):
        graphics_view.clearMask()
        img = Image.open(img_path)
        img = img.convert("RGB")  # 将图像转化成RGB的
        img = numpy.array(img)
        (img_height, img_width, _) = img.shape
        show_img = QImage(img, img_width, img_height, img_width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(show_img)  # 作为QPixmap对象，可以加载到一个空间中，通常是标签或者按钮中显示图像
        pixmapItem = QGraphicsPixmapItem(pixmap)  # 创建像素图元
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(pixmapItem)  # 将像素像元添加至场景
        graphics_view.setScene(scene)  # 将场景添加至视图
        graphics_view.fitInView(0, 0, img_width, img_height)  # 默认自动缩放至GraphicsView的大小
        del scene

    def start_retrieval(self):
        print('start_retrieval() clicked!')
        # 1.图片推理
        model = VGGNet()
        query_vector = model.vgg_extract_feat(self.retrieval_data_dir)  # 利用model.vgg_extract_feat()函数生成待查询图像的特征向量
        # 2.图片检索
        self.retrieval_DB_dir = self.lineEdit_retrieval_DB_dir.text()
        re = RetrievalEngine(self.retrieval_DB_dir, self.db_name)  # 传入参数 index_file, db_name
        print(self.comboBox.currentText())
        result = re.get_method(self.comboBox.currentText())(query_vector)  # 根据所选择的图像检索方式进行检索

        pic_1 = RetrievalResult()
        pic_2 = RetrievalResult()
        pic_3 = RetrievalResult()

        self.graphicsView_retrieved_img_1.clearMask()
        self.graphicsView_retrieved_img_2.clearMask()
        self.graphicsView_retrieved_img_3.clearMask()
        try:
            pic_1.name = result[0]['name'].decode('ascii')
            pic_1.full_name = os.path.join(self.train_data_dir, pic_1.name)
            pic_1.score = result[0]['score']
            self.label_retrieved_img_1.setText(str(pic_1.score) + '|' + pic_1.name)
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_1, pic_1.full_name)
        except:
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_1, 'cross.png')
            self.label_retrieved_img_1.setText('无结果')
        try:
            pic_2.name = result[1]['name'].decode('ascii')
            pic_2.full_name = os.path.join(self.train_data_dir, pic_2.name)
            pic_2.score = result[1]['score']
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_2, pic_2.full_name)
            self.label_retrieved_img_2.setText(str(pic_2.score) + '|' + pic_2.name)
        except:
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_2, 'cross.png')
            self.label_retrieved_img_2.setText('无结果')
        try:
            pic_3.name = result[2]['name'].decode('ascii')
            pic_3.full_name = os.path.join(self.train_data_dir, pic_3.name)
            pic_3.score = result[2]['score']
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_3, pic_3.full_name)
            self.label_retrieved_img_3.setText(str(pic_3.score) + '|' + pic_3.name)
        except:
            self.show_img_in_graphicview(self.graphicsView_retrieved_img_3, 'cross.png')
            self.label_retrieved_img_3.setText('无结果')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())  # 设置Pyqt深色系的样式表
    retrieval_program_window = RetrievalProgram()  # 类的实例化
    retrieval_program_window.show()  # 调用RetrievalProgram中show()方法
    sys.exit(app.exec_())
