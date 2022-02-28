# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'image_retrieval_widget.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1042, 800)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 800))
        font = QtGui.QFont()
        font.setPointSize(11)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_get_img_dir = QtWidgets.QPushButton(self.tab)
        self.pushButton_get_img_dir.setObjectName("pushButton_get_img_dir")
        self.verticalLayout_2.addWidget(self.pushButton_get_img_dir)
        self.pushButton_start_index = QtWidgets.QPushButton(self.tab)
        self.pushButton_start_index.setObjectName("pushButton_start_index")
        self.verticalLayout_2.addWidget(self.pushButton_start_index)
        self.pushButton_pause_and_save = QtWidgets.QPushButton(self.tab)
        self.pushButton_pause_and_save.setObjectName("pushButton_pause_and_save")
        self.verticalLayout_2.addWidget(self.pushButton_pause_and_save)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.tableWidget = QtWidgets.QTableWidget(self.tab)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.horizontalLayout.addWidget(self.tableWidget)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 5)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.pushButton_get_retrieval_img = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_get_retrieval_img.setObjectName("pushButton_get_retrieval_img")
        self.verticalLayout_9.addWidget(self.pushButton_get_retrieval_img)
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.toolButton_get_retrieval_file_dir = QtWidgets.QToolButton(self.groupBox_3)
        self.toolButton_get_retrieval_file_dir.setObjectName("toolButton_get_retrieval_file_dir")
        self.horizontalLayout_5.addWidget(self.toolButton_get_retrieval_file_dir)
        self.lineEdit_retrieval_DB_dir = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_retrieval_DB_dir.setObjectName("lineEdit_retrieval_DB_dir")
        self.horizontalLayout_5.addWidget(self.lineEdit_retrieval_DB_dir)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.comboBox = QtWidgets.QComboBox(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_2.addWidget(self.comboBox)
        self.horizontalLayout_2.setStretch(1, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_9.addWidget(self.groupBox_3)
        self.pushButton_start_retrieval = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_start_retrieval.setObjectName("pushButton_start_retrieval")
        self.verticalLayout_9.addWidget(self.pushButton_start_retrieval)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_9.addItem(spacerItem1)
        self.verticalLayout_9.setStretch(0, 1)
        self.verticalLayout_9.setStretch(1, 1)
        self.verticalLayout_9.setStretch(2, 1)
        self.verticalLayout_9.setStretch(3, 5)
        self.horizontalLayout_6.addLayout(self.verticalLayout_9)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(199, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.pushButton_previous_pic = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_previous_pic.sizePolicy().hasHeightForWidth())
        self.pushButton_previous_pic.setSizePolicy(sizePolicy)
        self.pushButton_previous_pic.setMouseTracking(False)
        self.pushButton_previous_pic.setObjectName("pushButton_previous_pic")
        self.horizontalLayout_3.addWidget(self.pushButton_previous_pic)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.graphicsView_retrieval_img = QtWidgets.QGraphicsView(self.groupBox)
        self.graphicsView_retrieval_img.setAcceptDrops(True)
        self.graphicsView_retrieval_img.setObjectName("graphicsView_retrieval_img")
        self.verticalLayout_5.addWidget(self.graphicsView_retrieval_img)
        self.label_retrieval_img = QtWidgets.QLabel(self.groupBox)
        self.label_retrieval_img.setAlignment(QtCore.Qt.AlignCenter)
        self.label_retrieval_img.setObjectName("label_retrieval_img")
        self.verticalLayout_5.addWidget(self.label_retrieval_img)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.pushButton_next_pic = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_next_pic.setObjectName("pushButton_next_pic")
        self.horizontalLayout_3.addWidget(self.pushButton_next_pic)
        spacerItem3 = QtWidgets.QSpacerItem(199, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 2)
        self.horizontalLayout_3.setStretch(3, 1)
        self.horizontalLayout_3.setStretch(4, 1)
        self.verticalLayout_4.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.graphicsView_retrieved_img_1 = QtWidgets.QGraphicsView(self.groupBox_2)
        self.graphicsView_retrieved_img_1.setObjectName("graphicsView_retrieved_img_1")
        self.verticalLayout_6.addWidget(self.graphicsView_retrieved_img_1)
        self.label_retrieved_img_1 = QtWidgets.QLabel(self.groupBox_2)
        self.label_retrieved_img_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_retrieved_img_1.setObjectName("label_retrieved_img_1")
        self.verticalLayout_6.addWidget(self.label_retrieved_img_1)
        self.horizontalLayout_4.addLayout(self.verticalLayout_6)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.graphicsView_retrieved_img_2 = QtWidgets.QGraphicsView(self.groupBox_2)
        self.graphicsView_retrieved_img_2.setObjectName("graphicsView_retrieved_img_2")
        self.verticalLayout_7.addWidget(self.graphicsView_retrieved_img_2)
        self.label_retrieved_img_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_retrieved_img_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_retrieved_img_2.setObjectName("label_retrieved_img_2")
        self.verticalLayout_7.addWidget(self.label_retrieved_img_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.graphicsView_retrieved_img_3 = QtWidgets.QGraphicsView(self.groupBox_2)
        self.graphicsView_retrieved_img_3.setObjectName("graphicsView_retrieved_img_3")
        self.verticalLayout_8.addWidget(self.graphicsView_retrieved_img_3)
        self.label_retrieved_img_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_retrieved_img_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_retrieved_img_3.setObjectName("label_retrieved_img_3")
        self.verticalLayout_8.addWidget(self.label_retrieved_img_3)
        self.horizontalLayout_4.addLayout(self.verticalLayout_8)
        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 1)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        self.verticalLayout_4.setStretch(0, 1)
        self.verticalLayout_4.setStretch(1, 1)
        self.horizontalLayout_6.addLayout(self.verticalLayout_4)
        self.horizontalLayout_6.setStretch(0, 1)
        self.horizontalLayout_6.setStretch(1, 5)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1042, 19))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionopen = QtWidgets.QAction(MainWindow)
        self.actionopen.setObjectName("actionopen")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.menu.addAction(self.actionopen)
        self.menu.addAction(self.actionOpen)
        self.menu.addSeparator()
        self.menu.addAction(self.actionSave)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.label.setBuddy(self.comboBox)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.pushButton_get_img_dir.clicked.connect(MainWindow.get_img_dir)
        self.pushButton_start_index.clicked.connect(MainWindow.start_index)
        self.pushButton_pause_and_save.clicked.connect(MainWindow.pause_and_save)
        self.pushButton_get_retrieval_img.clicked.connect(MainWindow.get_retrieval_img)
        self.pushButton_start_retrieval.clicked.connect(MainWindow.start_retrieval)
        self.toolButton_get_retrieval_file_dir.clicked.connect(MainWindow.get_retrieval_DB_dir)
        self.pushButton_previous_pic.clicked.connect(MainWindow.previous_pic)
        self.pushButton_next_pic.clicked.connect(MainWindow.next_pic)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Retrieval Program"))
        self.pushButton_get_img_dir.setText(_translate("MainWindow", "获得图像路径"))
        self.pushButton_start_index.setText(_translate("MainWindow", "开始"))
        self.pushButton_pause_and_save.setText(_translate("MainWindow", "终止并保存"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "特征提取"))
        self.pushButton_get_retrieval_img.setText(_translate("MainWindow", "导入图像"))
        self.groupBox_3.setTitle(_translate("MainWindow", "设置"))
        self.toolButton_get_retrieval_file_dir.setText(_translate("MainWindow", "库路径"))
        self.label.setText(_translate("MainWindow", "搜索方法"))
        self.comboBox.setItemText(0, _translate("MainWindow", "faiss"))
        self.comboBox.setItemText(1, _translate("MainWindow", "numpy"))
        self.comboBox.setItemText(2, _translate("MainWindow", "milves"))
        self.comboBox.setItemText(3, _translate("MainWindow", "es"))
        self.pushButton_start_retrieval.setText(_translate("MainWindow", "开始搜索"))
        self.groupBox.setTitle(_translate("MainWindow", "待搜索图像"))
        self.pushButton_previous_pic.setText(_translate("MainWindow", "上一个"))
        self.pushButton_previous_pic.setShortcut(_translate("MainWindow", "Left"))
        self.label_retrieval_img.setText(_translate("MainWindow", "待搜索图像"))
        self.pushButton_next_pic.setText(_translate("MainWindow", "下一个"))
        self.pushButton_next_pic.setShortcut(_translate("MainWindow", "Right"))
        self.groupBox_2.setTitle(_translate("MainWindow", "图像搜索结果"))
        self.label_retrieved_img_1.setText(_translate("MainWindow", "结果 1"))
        self.label_retrieved_img_2.setText(_translate("MainWindow", "结果 2"))
        self.label_retrieved_img_3.setText(_translate("MainWindow", "结果 3"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "图像搜索"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "编辑"))
        self.menu_3.setTitle(_translate("MainWindow", "设置"))
        self.menu_4.setTitle(_translate("MainWindow", "工具"))
        self.menu_5.setTitle(_translate("MainWindow", "帮助"))
        self.actionopen.setText(_translate("MainWindow", "New"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
