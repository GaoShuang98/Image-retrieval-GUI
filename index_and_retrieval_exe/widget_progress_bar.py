# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widget_progress_bar.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_widget_progress_bar(QtWidgets.QWidget):
    def __init__(self):
        super(Ui_widget_progress_bar, self).__init__()
        self.setupUi(self)

    def setupUi(self, widget_progress_bar):
        widget_progress_bar.setObjectName("widget_progress_bar")
        widget_progress_bar.resize(500, 200)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(widget_progress_bar.sizePolicy().hasHeightForWidth())
        widget_progress_bar.setSizePolicy(sizePolicy)
        widget_progress_bar.setMinimumSize(QtCore.QSize(500, 200))
        widget_progress_bar.setMaximumSize(QtCore.QSize(500, 200))
        widget_progress_bar.setStyleSheet("font: 10pt \"黑体\";")
        self.verticalLayout = QtWidgets.QVBoxLayout(widget_progress_bar)
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.feature_extract_info_label = QtWidgets.QLabel(widget_progress_bar)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.feature_extract_info_label.setFont(font)
        self.feature_extract_info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.feature_extract_info_label.setObjectName("feature_extract_info_label")
        self.verticalLayout.addWidget(self.feature_extract_info_label)
        self.progressBar = QtWidgets.QProgressBar(widget_progress_bar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setInputMethodHints(QtCore.Qt.ImhNone)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)

        self.retranslateUi(widget_progress_bar)
        QtCore.QMetaObject.connectSlotsByName(widget_progress_bar)

    def retranslateUi(self, widget_progress_bar):
        _translate = QtCore.QCoreApplication.translate
        widget_progress_bar.setWindowTitle(_translate("widget_progress_bar", "Form"))
        self.feature_extract_info_label.setText(_translate("widget_progress_bar", "Feature extract info ..."))
