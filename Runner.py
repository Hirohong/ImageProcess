import numpy
import sys
import cv2
import math
import numpy as np
import os.path
from os import path
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, QStandardPaths,QSize
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PIL import Image, ImageFont, ImageDraw
from threading import Timer
import time
from imageProcessor import ImageProcessor

is_print = True


def convertToRGB(value):
    if is_print:
        print(f"当前方法名：{sys._getframe().f_code.co_name}")
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


class DataModel():
    def __init__(self, width, height):
        if is_print:
            print(f'当前类名称：{self.__class__.__name__}')
        self.width = width
        self.height = height

    def getWidth(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        return self.width

    def getHeight(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        return self.height


class Worker(QThread):
    signal = pyqtSignal("PyQt_PyObject")

    def setTmApplication(self, tmApplication):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.tmApplication = tmApplication

    def setOperation(self, operation):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.operation = operation

    def setRadius(self, radius):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.radius = radius

    def setOperationString(self, operationString):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.operationString = operationString

    def run(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
            print(f'self.operation {self.operation}')
        if self.operation == 0:
            pass
        elif self.operation == 1:  # 灰度化
            print(f'01')
            self.tmApplication.imageProcessor.performGrayScale()
        elif self.operation == 2:  # 亮度
            print(f'02')
            self.tmApplication.imageProcessor.performBrightnessOperation(self.tmApplication.brightnessFactor)
        elif self.operation == 3:  # 对比度
            print(f'03')
            self.tmApplication.imageProcessor.performContrastOperation(self.tmApplication.contrastFactor)
        elif self.operation == 4:  # 裁切预览
            print(f'04')
            self.tmApplication.imageProcessor.getPreviewForCrop(self.tmApplication.startingPreviewX,
                                                                self.tmApplication.startingPreviewY,
                                                                self.tmApplication.endingPreviewX,
                                                                self.tmApplication.endingPreviewY)
        elif self.operation == 5:  # 裁切
            print(f'05')
            self.tmApplication.imageProcessor.performCrop(self.tmApplication.startingPreviewX,
                                                          self.tmApplication.startingPreviewY,
                                                          self.tmApplication.endingPreviewX,
                                                          self.tmApplication.endingPreviewY)
        elif self.operation == 6:  # 水平翻转
            print(f'06')
            self.tmApplication.imageProcessor.performHorizontalFlip()
        elif self.operation == 7:  # 垂直翻转
            print(f'07')
            self.tmApplication.imageProcessor.performVerticalFlip()
        elif self.operation == 8:  # 顺时针旋转90°
            # rotation clockwise 90
            print(f'08')
            self.tmApplication.imageProcessor.performClockwise()
        elif self.operation == 9:  # 逆时针旋转90°
            # rotation anticlockwise 90
            print(f'09')
            self.tmApplication.imageProcessor.performAnticlockwise()
        elif self.operation == 10:  # 旋转1800°
            # rotation 180
            print(f'10')
            self.tmApplication.imageProcessor.perform180()
        elif self.operation == 11:  # 覆盖
            # overlay
            print(f'11')
            self.tmApplication.imageProcessor.performOverlay(self.tmApplication.startingOverlayX,
                                                             self.tmApplication.startingOverlayY)
        elif self.operation == 12:  # 改变大小
            # resize
            print(f'12')
            self.tmApplication.imageProcessor.performResize(self.tmApplication.destHeight, self.tmApplication.destWidth)
        elif self.operation == 13:  # 高斯模糊
            # gaussian blur
            print(f'13')
            self.tmApplication.imageProcessor.performGuassianBlur(self.radius)
        elif self.operation == 14:  # 水平motion模糊
            # vertical motion blur
            print(f'14')
            self.tmApplication.imageProcessor.performVerticalMotionBlur(self.radius)
        elif self.operation == 15:  # 垂直motion模糊
            # horizontal motion blur
            print(f'15')
            self.tmApplication.imageProcessor.performHorizontalMotionBlur(self.radius)
        elif self.operation == 16:  # box模糊
            # box blur
            print(f'16')
            self.tmApplication.imageProcessor.performBoxBlur()
        elif self.operation == 17:  # 反色
            # invert
            print(f'17')
            self.tmApplication.imageProcessor.performInvert()
        elif self.operation == 18:  # 锐化
            # sharpen
            print(f'18')
            self.tmApplication.imageProcessor.performSharpening()
        elif self.operation == 19:  # 反锐化
            # unsharp
            print(f'19')
            self.tmApplication.imageProcessor.performUnsharpening()
        elif self.operation == 20:  # 拉普拉斯
            # laplace
            print(f'20')
            self.tmApplication.imageProcessor.performLaplace()
        elif self.operation == 21:  # 添加文本
            # put text on image
            print(f'21')
            self.tmApplication.imageProcessor.performTextOperation(self.tmApplication.textOnImage,
                                                                   self.tmApplication.fontPath,
                                                                   self.tmApplication.fontColorCode,
                                                                   self.tmApplication.fontSize,
                                                                   self.tmApplication.textX, self.tmApplication.textY)
        elif self.operation == 22:  # zoom in
            # perform zoom in
            print(f'22')
            self.tmApplication.imageProcessor.performZoomIn()
        elif self.operation == 23:  # zoom out
            # perform zoom out
            print(f'23')
            self.tmApplication.imageProcessor.performZoomOut()
        elif self.operation == 24:  # 添加图片水印
            # watermark image operation
            print(f'24')
            self.tmApplication.imageProcessor.performImageWatermark(self.tmApplication.watermarkImagePath,
                                                                    self.tmApplication.watermarkImageOpacity,
                                                                    self.tmApplication.watermarkImageCoordinates)
        elif self.operation == 25:  # 添加文字水印
            # watermark text operation
            print(f'25')
            self.tmApplication.imageProcessor.performTextWatermark(self.tmApplication.watermarkText,
                                                                   self.tmApplication.watermarkTextSize,
                                                                   self.tmApplication.watermarkTextThickness,
                                                                   self.tmApplication.watermarkTextOpacity,
                                                                   self.tmApplication.watermarkTextCoordinates)
        elif self.operation == 26:  # 边框
            # normal border
            print(f'26')
            self.tmApplication.imageProcessor.performNormalBorderOperation(self.operationString)
        elif self.operation == 27:  # 重复边框
            # replicate border
            print(f'27')
            self.tmApplication.imageProcessor.performReplicateBorderOperation(self.operationString)
            # 变换
        elif self.operation == 29:  # 傅里叶变换
            print(f'29')
            self.tmApplication.imageProcessor.fourierTransform()
        elif self.operation == 30:  # 离散余弦变换
            print(f'30')
            self.tmApplication.imageProcessor.discreteCosineTransform()
        elif self.operation == 31:  # Radom变换
            print(f'31')
            self.tmApplication.imageProcessor.discreteRadonTransform()
        # 噪声
        elif self.operation == 32:  # 高斯噪声
            print(f'32')
            self.tmApplication.imageProcessor.gaussianNoiseImage()
        elif self.operation == 33:  # 椒盐噪声
            print(f'33')
            self.tmApplication.imageProcessor.saltPepperImage()
        # elif self.operation == 34:# 斑点噪声
        #    self.tmApplication.imageProcessor
        # elif self.operation == 35:# 泊松噪声
        #    self.tmApplication.imageProcessor
        # 滤波
        elif self.operation == 36:  # 高通滤波
            print(f'36')
            self.tmApplication.imageProcessor.highPassFilter()
        elif self.operation == 37:  # 低通滤波
            print(f'37')
            self.tmApplication.imageProcessor.medianBlur()
        elif self.operation == 38:  # 平滑滤波（线性）  只有一个平滑滤波 不知是线性还是非线性
            print(f'38')
            self.tmApplication.imageProcessor.blur()
        # elif self.operation == 39:# 平滑滤波（非线性）
        #    self.tmApplication.imageProcessor
        elif self.operation == 40:  # 锐化滤波（线性）  只有一个锐化滤波 不知是线性还是非线性
            print(f'40')
            self.tmApplication.imageProcessor.bilateralFilter()
        # elif self.operation == 41:# 锐化滤波（非线性）
        #    self.tmApplication.imageProcessor
        # 直方图统计
        # elif self.operation == 42:# R直方图
        #    self.tmApplication.imageProcessor
        # elif self.operation == 43:# G直方图
        #    self.tmApplication.imageProcessor
        # elif self.operation == 44:# B直方图
        #    self.tmApplication.imageProcessor
        # 有颜色直方图和灰度直方图
        elif self.operation == 42:  # 颜色直方图
            print(f'42')
            self.tmApplication.imageProcessor.pixelFrequencyHistogram()
        elif self.operation == 43:  # 灰度直方图
            print(f'43')
            self.tmApplication.imageProcessor.grayFrequencyHistogram()
        # 图像增强
        elif self.operation == 45:  # 伪彩色增强
            print(f'45')
            self.tmApplication.imageProcessor.pseudoColorEnhancement()
        elif self.operation == 46:  # 真彩色增强
            print(f'46')
            self.tmApplication.imageProcessor.trueColorEnhancement()
        elif self.operation == 47:  # 直方图均衡
            print(f'47')
            self.tmApplication.imageProcessor.histNormalized()
        elif self.operation == 48:  # NTSC颜色模型
            print(f'48')
            self.tmApplication.imageProcessor.bGR2NTSC()
        elif self.operation == 49:  # YCbCr颜色模型
            print(f'49')
            self.tmApplication.imageProcessor.bGR2YCR_CB()
        elif self.operation == 50:  # HSV颜色模型
            print(f'50')
            self.tmApplication.imageProcessor.bGR2HSV()

        elif self.operation == 51:  # 阈值分割 otsu
            print(f'51')
            self.tmApplication.imageProcessor.otsuBinarization()
        elif self.operation == 56:  # 阈值分割 二值
            print(f'56')
            self.tmApplication.imageProcessor.binalization()
        elif self.operation == 52:  # 形态学处理 膨胀
            print(f'52')
            self.tmApplication.imageProcessor.dilateImage()
        elif self.operation == 53:  # 形态学处理 腐蚀
            print(f'53')
            self.tmApplication.imageProcessor.erodeImage()
        elif self.operation == 54:  # 特征提取  角点检测
            print(f'54')
            self.tmApplication.imageProcessor.featureExtraction()
        elif self.operation == 57:  # 特征提取  canny边缘检测
            self.tmApplication.imageProcessor.cannyEdgeDetection(self.tmApplication.threshold1, self.tmApplication.threshold2)
            # self.tmApplication.imageProcessor.cannyEdgeDetection(self.threshold1, self.threshold2)
            # self.tmApplication.imageProcessor.cannyEdgeDetection()
        elif self.operation == 55:  # 图像分类与识别
            print(f'59')
            self.tmApplication.imageProcessor.classification()
        elif self.operation == 58:  # 显示原图
            print(f'58')
            self.tmApplication.imageProcessor.getOriginalImage()
        print(f'work.run end')
        self.signal.emit("Completed")


class TMApplication(QMainWindow, QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        if is_print:
            print(f'当前类名称：{self.__class__.__name__}')
        self.brightnessFactor = 0
        self.contrastFactor = 0
        self.worker = Worker()
        self.images = {}
        self.model = model
        # self.imageProcessor = self.ImageProcessor(self)
        self.imageProcessor = ImageProcessor(self)
        self.setWindowTitle("Pixi")
        self.setFixedSize(self.model.getWidth(), self.model.getHeight())
        self.setWindowIcon(QtGui.QIcon("img/logo_black.png"))
        self.basewidget = QWidget()
        self.leftwidget = QWidget()  # 基础窗口控件
        self.scrollArea = QScrollArea()  # 出现滚动条
        self.widget = QWidget()  # 基础窗口控件
        self.leftPanel = QLabel(self.basewidget)

        # self.basewidget.setMaximumWidth(1046)
        self.setCentralWidget(self.basewidget)
        self.baselayout = QHBoxLayout(self.basewidget)

        # self.leftwidget.setGeometry(0, 0, 38, 38 * 11)
        self.baselayout.setContentsMargins(0, 0, 0, 0)
        self.baselayout.addWidget(self.leftwidget, 4)

        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setMaximumWidth(1000)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.widget)

        self.baselayout.addWidget(self.scrollArea, 126)
        self.createMenuBar()
        self.createToolBar()
        self.createLeftToolBar()
        self.createStatusBar()
        self.loadFonts()
        self.loadLogFiles()
        self.setFrameGeometry()


    def setFrameGeometry(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        frameGeometry = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()  # 居中窗口
        frameGeometry.moveCenter(centerPoint)
        self.move(frameGeometry.topLeft())  # 改编窗口部件的位置

    def loadLogFiles(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.logFiles = list()
        self.logFiles.append("Loading font database...")
        self.logFiles.append("Loading menu...")
        self.logFiles.append("Loading toolbar...")
        self.logFiles.append("Loading image processor...")
        self.logFiles.append("Loading status bar...")
        self.logFiles.append("Loading log files...")
        self.logFiles.append("Loading cache...")
        self.logFiles.append("Loading worker thread...")
        self.logFiles.append("Loading panels...")
        self.logFiles.append("Initializing UI...")

    def loadFonts(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.family_to_path = dict()
        self.accounted = list()
        self.unloadable = list()
        font_paths = QStandardPaths.standardLocations(QStandardPaths.FontsLocation)
        font_database = QFontDatabase()
        for path in font_paths:
            for font in os.listdir(path):
                font_path = os.path.join(path, font)
                index = font_database.addApplicationFont(font_path)
                if index < 0: self.unloadable.append(font_path)
                font_names = font_database.applicationFontFamilies(index)
                for font_name in font_names:
                    if font_name in self.family_to_path:
                        self.accounted.append((font_name, font_path))
                    else:
                        self.family_to_path[font_name] = font_path

    def checkPreviousStuff(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if path.isfile("temp.png"):
            os.remove("temp.png")
        elif path.isfile("temp.jpg"):
            os.remove("temp.jpg")

    def updateStatusBar(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        msg = self.images["status"]
        self.statusBar.showMessage(msg, 3000)

    # 显示照片，照片信息&上一部和下一步操作可用
    def updateImageOnUI(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        imagePath = self.images["filterImageName"]
        if imagePath == "": imagePath = self.images["path"]
        imageName = self.images["name"]
        imageResolution = str(self.images["height"]) + "x" + str(self.images["width"])
        imageSize = str(self.images["size"]) + " MB"

        # label for image
        self.lx1 = 1
        self.ly1 = 61
        self.lx2 = 828
        self.ly2 = 508
        self.rx1 = self.lx1 + 1 + self.lx2 + 219
        self.ry1 = self.ly1
        self.rx2 = 199
        self.ry2 = 18


        self.leftPanel.setPixmap(QPixmap(imagePath))

        # self.leftPanel.setGeometry(self.lx1,self.ly1,self.lx2,self.ly2)
        # self.leftPanel.setStyleSheet("border: 1px solid black; width:200px;")

        # self.scrollArea = QScrollArea()  # 出现滚动条
        # self.widget = QWidget(self.basewidget)  # 基础窗口控件
        # self.baselayout.addWidget(self.scrollArea)
        self.boxLayout = QVBoxLayout()  # 设置垂直布局

        self.boxLayout.addWidget(self.leftPanel)
        self.widget.setLayout(self.boxLayout)

        # self.setCentralWidget(self.scrollArea)
        # 显示照片信息
        self.rightPanel1 = QLabel(self.basewidget)
        self.rightPanel1.setGeometry(self.rx1, self.ry1, self.rx2, self.ry2)
        self.rightPanel1.setText("Image Details")
        self.rightPanel1.setAlignment(QtCore.Qt.AlignCenter)
        self.rightPanel1.setStyleSheet("background-color:white;")
        self.rightPanel1.show()

        self.rightPanel1 = QLabel(self.basewidget)
        self.rightPanel1.setGeometry(self.rx1, self.ry1 + 18, self.rx2, self.ry2)
        self.rightPanel1.setAlignment(QtCore.Qt.AlignLeft)
        self.rightPanel1.setStyleSheet("background-color:white; padding-left:10px;")
        self.rightPanel1.show()

        self.rightPanel1 = QLabel(self.basewidget)
        self.rightPanel1.setGeometry(self.rx1, self.ry1 + 18 + 18, self.rx2, self.ry2)
        self.rightPanel1.setText("File: " + imageName)
        self.rightPanel1.setAlignment(QtCore.Qt.AlignLeft)
        self.rightPanel1.setStyleSheet("background-color:white; padding-left:10px;")
        self.rightPanel1.show()

        self.rightPanel1 = QLabel(self.basewidget)
        self.rightPanel1.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18, self.rx2, self.ry2)
        self.rightPanel1.setText("Resolution: " + imageResolution)
        self.rightPanel1.setAlignment(QtCore.Qt.AlignLeft)
        self.rightPanel1.setStyleSheet("background-color:white; padding-left:10px;")
        self.rightPanel1.show()

        self.rightPanel1 = QLabel(self.basewidget)
        self.rightPanel1.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18, self.rx2, self.ry2)
        self.rightPanel1.setText("Size: " + imageSize)
        self.rightPanel1.setAlignment(QtCore.Qt.AlignLeft)
        self.rightPanel1.setStyleSheet("background-color:white; padding-left:10px;")
        self.rightPanel1.show()
        self.updateStacks()

    # 依据条件将上一张和下一张操作可用
    def updateStacks(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if len(self.images["undoStack"]) > 1:
            self.undoAct.setEnabled(True)
        else:
            self.undoAct.setEnabled(False)

        if len(self.images["redoStack"]) > 0:
            self.redoAct.setEnabled(True)
        else:
            self.redoAct.setEnabled(False)

    def showFontSelectorDialog(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        font, ok = QFontDialog.getFont()
        if ok:
            fontDetails = font.toString().split(',')
            self.fontName = fontDetails[0]
            self.fontSize = fontDetails[1]
            self.fontType = fontDetails[-1]
            if len(self.fontName) > 15:
                self.fontName = self.fontName[:15]
            self.textPanelFontButton.setText(self.fontName)
            self.textPanelFontSizeLabel.setText(self.fontSize)
            try:
                self.fontPath = self.family_to_path[self.fontName]
            except:
                errorMessage = QtWidgets.QErrorMessage(self)
                errorMessage.showMessage(f"The specified font: {self.fontName} is not available !")
                errorMessage.setWindowTitle(f"Font unavailable")

    def showColorSelectorDialog(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        color = QColorDialog.getColor()
        if color.isValid():
            self.fontColorCode = color.name()
            self.textPanelColorButton.setText(self.fontColorCode)

    def updateTextPanelParameters(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.textOnImage = self.textPanelTextArea.text()
        self.textX = self.textPanelXCoordinate.text()
        self.textY = self.textPanelYCoordinate.text()
        if len(self.textX) > 0 and len(self.textY) > 0:
            self.textX = int(self.textX)
            self.textY = int(self.textY)

    def applyText(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(21)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def setOpacitySliderValue(self, sliderNo):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if sliderNo == 1:
            opacitySliderValue = str(self.watermarkPanelImageOpacitySlider.value())
            self.watermarkPanelImageOpacitySliderValueLabel.setText(opacitySliderValue)
        else:
            opacitySliderValue = str(self.watermarkPanelTextOpacitySlider.value())
            self.watermarkPanelTextOpacitySliderValueLabel.setText(opacitySliderValue)

    def showWatermarkPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.watermarkImagePath = ""
        self.watermarkPanelImageRadioButton.setChecked(True)
        self.watermarkPanelTitle.show()
        self.watermarkPanelImageRadioButton.show()
        self.watermarkPanelTextRadioButton.show()
        self.watermarkPanelImageLabel.show()
        self.watermarkPanelImageButton.show()
        self.watermarkPanelXImageTitleLabel.show()
        self.watermarkPanelXImageTextArea.show()
        self.watermarkPanelYImageTitleLabel.show()
        self.watermarkPanelYImageTextArea.show()
        self.watermarkPanelImageOpacityPanel.show()
        self.watermarkPanelImageOpacitySlider.show()
        self.watermarkPanelImageOpacitySliderValueLabel.show()
        self.watermarkPanelImageOKButton.show()
        self.watermarkPanelImageCloseButton.show()

    def hideWatermarkPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.watermarkPanelTitle.hide()
        self.watermarkPanelImageRadioButton.hide()
        self.watermarkPanelTextRadioButton.hide()
        self.hideWatermarkImagePanel()
        self.hideWatermarkTextPanel()

    def showWatermarkImagePanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.hideWatermarkTextPanel()
        self.watermarkPanelImageRadioButton.show()
        self.watermarkPanelTextRadioButton.show()
        self.watermarkPanelImageLabel.show()
        self.watermarkPanelImageButton.show()
        self.watermarkPanelXImageTitleLabel.show()
        self.watermarkPanelXImageTextArea.show()
        self.watermarkPanelYImageTitleLabel.show()
        self.watermarkPanelYImageTextArea.show()
        self.watermarkPanelImageOpacityPanel.show()
        self.watermarkPanelImageOpacitySlider.show()
        self.watermarkPanelImageOpacitySliderValueLabel.show()
        self.watermarkPanelImageOKButton.show()
        self.watermarkPanelImageCloseButton.show()

    def hideWatermarkImagePanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.watermarkPanelImageLabel.hide()
        self.watermarkPanelImageButton.hide()
        self.watermarkPanelImageButton.setText("Choose...")
        self.watermarkPanelXImageTitleLabel.hide()
        self.watermarkPanelXImageTextArea.hide()
        self.watermarkPanelXImageTextArea.setText("0")
        self.watermarkPanelYImageTitleLabel.hide()
        self.watermarkPanelYImageTextArea.hide()
        self.watermarkPanelYImageTextArea.setText("0")
        self.watermarkPanelImageOpacityPanel.hide()
        self.watermarkPanelImageOpacitySlider.hide()
        self.watermarkPanelImageOpacitySlider.setFocusPolicy(Qt.NoFocus)
        self.watermarkPanelImageOpacitySlider.setTickPosition(QSlider.NoTicks)
        self.watermarkPanelImageOpacitySlider.setTickInterval(10)
        self.watermarkPanelImageOpacitySlider.setSingleStep(1)
        self.watermarkPanelImageOpacitySlider.setValue(50)
        self.watermarkPanelImageOpacitySliderValueLabel.hide()
        self.watermarkPanelImageOpacitySliderValueLabel.setText("50")
        self.watermarkPanelImageOKButton.hide()
        self.watermarkPanelImageCloseButton.hide()

    def hideWatermarkTextPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.watermarkPanelTextLabel.hide()
        self.watermarkPanelTextButton.hide()
        self.watermarkPanelTextButton.setText("")
        self.watermarkPanelXTextTitleLabel.hide()
        self.watermarkPanelXTextTextArea.hide()
        self.watermarkPanelXTextTextArea.setText("0")
        self.watermarkPanelYTextTitleLabel.hide()
        self.watermarkPanelYTextTextArea.hide()
        self.watermarkPanelYTextTextArea.setText("0")
        self.watermarkPanelTextThicknessLabel.hide()
        self.watermarkPanelTextThicknessTextArea.hide()
        self.watermarkPanelTextThicknessTextArea.setText("0")
        self.watermarkPanelTextSizeLabel.hide()
        self.watermarkPanelTextSizeTextArea.hide()
        self.watermarkPanelTextSizeTextArea.setText("0")
        self.watermarkPanelTextOpacityPanel.hide()
        self.watermarkPanelTextOpacitySlider.hide()
        self.watermarkPanelTextOpacitySlider.setFocusPolicy(Qt.NoFocus)
        self.watermarkPanelTextOpacitySlider.setTickPosition(QSlider.NoTicks)
        self.watermarkPanelTextOpacitySlider.setTickInterval(10)
        self.watermarkPanelTextOpacitySlider.setSingleStep(1)
        self.watermarkPanelTextOpacitySlider.setValue(50)
        self.watermarkPanelTextOpacitySliderValueLabel.hide()
        self.watermarkPanelTextOKButton.hide()
        self.watermarkPanelTextCloseButton.hide()

    def hideCannyPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.cannyPanelTitle.hide()
        self.cannyPanelThreshold1Label.hide()
        self.cannyPanelThreshold1Area.hide()
        self.cannyPanelThreshold2Label.hide()
        self.cannyPanelThreshold2Area.hide()
        self.cannyPanelOKButton.hide()
        self.cannyPanelCancelButton.hide()

    def showCannyPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.hideCropPanel()
        self.hideOverlayPanel()
        self.hideFlipPanel()
        self.hideRotatePanel()
        self.cannyPanelTitle.show()
        self.cannyPanelThreshold1Label.show()
        self.cannyPanelThreshold1Area.show()
        self.cannyPanelThreshold2Label.show()
        self.cannyPanelThreshold2Area.show()
        self.cannyPanelOKButton.show()
        self.cannyPanelCancelButton.show()

    def showWatermarkTextPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.hideWatermarkImagePanel()
        self.watermarkPanelTextLabel.show()
        self.watermarkPanelTextButton.show()
        self.watermarkPanelXTextTitleLabel.show()
        self.watermarkPanelXTextTextArea.show()
        self.watermarkPanelYTextTitleLabel.show()
        self.watermarkPanelYTextTextArea.show()
        self.watermarkPanelTextThicknessLabel.show()
        self.watermarkPanelTextThicknessTextArea.show()
        self.watermarkPanelTextSizeLabel.show()
        self.watermarkPanelTextSizeTextArea.show()
        self.watermarkPanelTextOpacityPanel.show()
        self.watermarkPanelTextOpacitySlider.show()
        self.watermarkPanelTextOpacitySliderValueLabel.show()
        self.watermarkPanelTextOKButton.show()
        self.watermarkPanelTextCloseButton.show()

    def openWatermarkImageChooser(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(self, "Choose an image", "",
                                                  "JPG (*.jpg);;PNG (*.png);;JPEG (*.jpeg);", options=options)
        fileName = ""
        if filePath:
            self.watermarkImagePath = filePath
            fileName = filePath[filePath.rfind("/") + 1:]
            if len(fileName) > 10:
                fn = fileName[:8] + "..."
                self.watermarkPanelImageButton.setText(fn)
            else:
                self.watermarkPanelImageButton.setText(fileName)

    def validateWatermarkImageParameters(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        hasError = False
        x = self.watermarkPanelXImageTextArea.text()
        y = self.watermarkPanelYImageTextArea.text()
        errorMessage = ""
        errorMessageDialog = QtWidgets.QErrorMessage(self)
        if len(self.watermarkImagePath) == 0:
            hasError = True
            errorMessage += "Image not selected. \r\n"
        if len(x) == 0 or int(x) == 0:
            errorMessage += "Invalid x-coordinte. \r\n"
            hasError = True
        if len(y) == 0 or int(y) == 0:
            errorMessage += "Invalid y-coordinte. \r\n"
            hasError = True
        if hasError:
            errorMessageDialog.showMessage(errorMessage)
            errorMessageDialog.setWindowTitle("Error !")
            errorMessageDialog.show()
        else:
            self.applyImageWatermark()

    def applyImageWatermark(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        x = int(self.watermarkPanelXImageTextArea.text())
        y = int(self.watermarkPanelYImageTextArea.text())
        self.watermarkImageCoordinates = x, y
        self.watermarkImageOpacity = int(self.watermarkPanelImageOpacitySliderValueLabel.text())
        self.worker.setTmApplication(self)
        self.worker.setOperation(24)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def applyTextWatermark(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        text = self.watermarkPanelTextButton.text()
        x = int(self.watermarkPanelXTextTextArea.text())
        y = int(self.watermarkPanelYTextTextArea.text())
        thickness = int(self.watermarkPanelTextThicknessTextArea.text())
        size = int(self.watermarkPanelTextSizeTextArea.text())
        self.watermarkText = text
        self.watermarkTextSize = size
        self.watermarkTextThickness = thickness
        self.watermarkTextCoordinates = x, y
        self.watermarkTextOpacity = int(self.watermarkPanelTextOpacitySliderValueLabel.text())
        self.worker.setTmApplication(self)
        self.worker.setOperation(25)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def validateWatermarkTextParameters(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        hasError = False
        text = self.watermarkPanelTextButton.text()
        x = self.watermarkPanelXTextTextArea.text()
        y = self.watermarkPanelYTextTextArea.text()
        thickness = self.watermarkPanelTextThicknessTextArea.text()
        size = self.watermarkPanelTextSizeTextArea.text()
        errorMessage = ""
        errorMessageDialog = QtWidgets.QErrorMessage(self)
        if len(text) == 0:
            errorMessage += "Invalid text input.\r\n"
            hasError = True
        if len(x) == 0 or int(x) == 0:
            errorMessage += "Invalid x-coordinte. \r\n"
            hasError = True
        if len(y) == 0 or int(y) == 0:
            errorMessage += "Invalid y-coordinte. \r\n"
            hasError = True
        if len(thickness) == 0 or int(thickness) == 0:
            errorMessage += "Invalid thickness. \r\n"
            hasError = True
        if len(size) == 0 or int(size) == 0:
            errorMessage += "Invalid text size. \r\n"
            hasError = True
        if hasError:
            errorMessageDialog.showMessage(errorMessage)
            errorMessageDialog.setWindowTitle("Error !")
            errorMessageDialog.show()
        else:
            self.applyTextWatermark()

    # 生成右侧操作面板
    def createPanels(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.lx1 = 1
        self.ly1 = 61
        self.lx2 = 828
        self.ly2 = 508
        self.rx1 = self.lx1 + 1 + self.lx2 + 219
        self.ry1 = self.ly1 + 90
        self.rx2 = 199
        self.ry2 = 18

        # watermarkPanel

        self.watermarkPanelTitle = QLabel(self)
        self.watermarkPanelTitle.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2, self.rx2, self.ry2 + 5)
        self.watermarkPanelTitle.setText("Watermark")
        self.watermarkPanelTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelTitle.hide()

        self.watermarkPanelImageRadioButton = QRadioButton(self)
        self.watermarkPanelImageRadioButton.setGeometry(self.rx1 + 30, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15,
                                                        self.rx2, self.ry2 + 5)
        self.watermarkPanelImageRadioButton.setText("Image")
        self.watermarkPanelImageRadioButton.setChecked(True)
        self.watermarkPanelImageRadioButton.clicked.connect(self.showWatermarkImagePanel)
        self.watermarkPanelImageRadioButton.hide()

        self.watermarkPanelTextRadioButton = QRadioButton(self)
        self.watermarkPanelTextRadioButton.setGeometry(self.rx1 + 120, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15,
                                                       self.rx2, self.ry2 + 5)
        self.watermarkPanelTextRadioButton.setText("Text")
        self.watermarkPanelTextRadioButton.clicked.connect(self.showWatermarkTextPanel)
        self.watermarkPanelTextRadioButton.hide()

        # watermarkTextPanel

        self.watermarkPanelTextLabel = QLabel(self)
        self.watermarkPanelTextLabel.setGeometry(self.rx1 + 10, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40,
                                                 45, self.ry2)
        self.watermarkPanelTextLabel.setText("Text : ")
        self.watermarkPanelTextLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelTextLabel.hide()

        self.watermarkPanelTextButton = QLineEdit("", self)
        self.watermarkPanelTextButton.setGeometry(self.rx1 + 60, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40,
                                                  85, self.ry2 + 5)
        self.watermarkPanelTextButton.hide()

        self.watermarkPanelXTextTitleLabel = QLabel(self)
        self.watermarkPanelXTextTitleLabel.setGeometry(self.rx1,
                                                       self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10,
                                                       45, self.ry2)
        self.watermarkPanelXTextTitleLabel.setText("X : ")
        self.watermarkPanelXTextTitleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelXTextTitleLabel.hide()

        self.watermarkPanelXTextTextArea = QLineEdit("0", self)
        self.watermarkPanelXTextTextArea.setGeometry(self.rx1 + 35,
                                                     self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10, 40,
                                                     self.ry2)
        self.watermarkPanelXTextTextArea.hide()

        self.watermarkPanelYTextTitleLabel = QLabel(self)
        self.watermarkPanelYTextTitleLabel.setGeometry(self.rx1 + 35 + 35 + 10,
                                                       self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10,
                                                       45, self.ry2)
        self.watermarkPanelYTextTitleLabel.setText("Y : ")
        self.watermarkPanelYTextTitleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelYTextTitleLabel.hide()

        self.watermarkPanelYTextTextArea = QLineEdit("0", self)
        self.watermarkPanelYTextTextArea.setGeometry(self.rx1 + 35 + 35 + 35 + 10,
                                                     self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10, 40,
                                                     self.ry2)
        self.watermarkPanelYTextTextArea.hide()

        self.watermarkPanelTextThicknessLabel = QLabel(self)
        self.watermarkPanelTextThicknessLabel.setGeometry(self.rx1 + 10,
                                                          self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10 + 35,
                                                          60, self.ry2)
        self.watermarkPanelTextThicknessLabel.setText("Thickness : ")
        self.watermarkPanelTextThicknessLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelTextThicknessLabel.hide()

        self.watermarkPanelTextThicknessTextArea = QLineEdit("0", self)
        self.watermarkPanelTextThicknessTextArea.setGeometry(self.rx1 + 15 + 55,
                                                             self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10 + 35,
                                                             40, self.ry2)
        self.watermarkPanelTextThicknessTextArea.hide()

        self.watermarkPanelTextSizeLabel = QLabel(self)
        self.watermarkPanelTextSizeLabel.setGeometry(self.rx1 + 15 + 55 + 40 + 5,
                                                     self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10 + 35,
                                                     40, self.ry2)
        self.watermarkPanelTextSizeLabel.setText("Size : ")
        self.watermarkPanelTextSizeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelTextSizeLabel.hide()

        self.watermarkPanelTextSizeTextArea = QLineEdit("0", self)
        self.watermarkPanelTextSizeTextArea.setGeometry(self.rx1 + 15 + 55 + 40 + 10 + 35,
                                                        self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10 + 35,
                                                        40, self.ry2)
        self.watermarkPanelTextSizeTextArea.hide()

        self.watermarkPanelTextOpacityPanel = QLabel("Opacity : ", self)
        self.watermarkPanelTextOpacityPanel.setGeometry(self.rx1 + 10,
                                                        self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 30 + 45,
                                                        50, self.ry2)
        self.watermarkPanelTextOpacityPanel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelTextOpacityPanel.hide()

        self.watermarkPanelTextOpacitySlider = QSlider(Qt.Horizontal, self)
        self.watermarkPanelTextOpacitySlider.setFocusPolicy(Qt.NoFocus)
        self.watermarkPanelTextOpacitySlider.setTickPosition(QSlider.NoTicks)
        self.watermarkPanelTextOpacitySlider.setTickInterval(10)
        self.watermarkPanelTextOpacitySlider.setSingleStep(1)
        self.watermarkPanelTextOpacitySlider.setValue(50)
        self.watermarkPanelTextOpacitySlider.setGeometry(self.rx1 + 10 + 55,
                                                         self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 60 + 15,
                                                         100, 20)
        self.watermarkPanelTextOpacitySlider.valueChanged.connect(lambda: self.setOpacitySliderValue(2))
        self.watermarkPanelTextOpacitySlider.hide()

        self.watermarkPanelTextOpacitySliderValueLabel = QLabel(self)
        self.watermarkPanelTextOpacitySliderValueLabel.setGeometry(self.rx1 + 10 + 145,
                                                                   self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 30 + 45,
                                                                   50, self.ry2)
        self.watermarkPanelTextOpacitySliderValueLabel.setText("50")
        self.watermarkPanelTextOpacitySliderValueLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelTextOpacitySliderValueLabel.hide()

        self.watermarkPanelTextOKButton = QPushButton("OK", self)
        self.watermarkPanelTextOKButton.setGeometry(self.rx1 + 10 + 15,
                                                    self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 30 + 20 + 60,
                                                    70, self.ry2 + 5)
        self.watermarkPanelTextOKButton.setIcon(QIcon("img/ok.png"))
        self.watermarkPanelTextOKButton.clicked.connect(self.validateWatermarkTextParameters)
        self.watermarkPanelTextOKButton.hide()

        self.watermarkPanelTextCloseButton = QPushButton("Close", self)
        self.watermarkPanelTextCloseButton.setGeometry(self.rx1 + 10 + 92,
                                                       self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 30 + 20 + 60,
                                                       70, self.ry2 + 5)
        self.watermarkPanelTextCloseButton.setIcon(QIcon("img/cancel.png"))
        self.watermarkPanelTextCloseButton.clicked.connect(self.hideWatermarkPanel)
        self.watermarkPanelTextCloseButton.hide()

        # watermarkImagePanel

        self.watermarkPanelImageLabel = QLabel(self)
        self.watermarkPanelImageLabel.setGeometry(self.rx1 + 10, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40,
                                                  45, self.ry2)
        self.watermarkPanelImageLabel.setText("Image : ")
        self.watermarkPanelImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelImageLabel.hide()

        self.watermarkPanelImageButton = QPushButton("Choose...", self)
        self.watermarkPanelImageButton.setGeometry(self.rx1 + 60, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40,
                                                   85, self.ry2 + 5)
        self.watermarkPanelImageButton.setIcon(QIcon("img/choose.png"))
        self.watermarkPanelImageButton.clicked.connect(self.openWatermarkImageChooser)
        self.watermarkPanelImageButton.hide()

        self.watermarkPanelXImageTitleLabel = QLabel(self)
        self.watermarkPanelXImageTitleLabel.setGeometry(self.rx1,
                                                        self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10,
                                                        45, self.ry2)
        self.watermarkPanelXImageTitleLabel.setText("X : ")
        self.watermarkPanelXImageTitleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelXImageTitleLabel.hide()

        self.watermarkPanelXImageTextArea = QLineEdit("0", self)
        self.watermarkPanelXImageTextArea.setGeometry(self.rx1 + 35,
                                                      self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10,
                                                      40, self.ry2)
        self.watermarkPanelXImageTextArea.hide()

        self.watermarkPanelYImageTitleLabel = QLabel(self)
        self.watermarkPanelYImageTitleLabel.setGeometry(self.rx1 + 35 + 35 + 10,
                                                        self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10,
                                                        45, self.ry2)
        self.watermarkPanelYImageTitleLabel.setText("Y : ")
        self.watermarkPanelYImageTitleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelYImageTitleLabel.hide()

        self.watermarkPanelYImageTextArea = QLineEdit("0", self)
        self.watermarkPanelYImageTextArea.setGeometry(self.rx1 + 35 + 35 + 35 + 10,
                                                      self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 10,
                                                      40, self.ry2)
        self.watermarkPanelYImageTextArea.hide()

        self.watermarkPanelImageOpacityPanel = QLabel(self)
        self.watermarkPanelImageOpacityPanel.setGeometry(self.rx1 + 10,
                                                         self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 30 + 20,
                                                         50, self.ry2)
        self.watermarkPanelImageOpacityPanel.setText("Opacity : ")
        self.watermarkPanelImageOpacityPanel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelImageOpacityPanel.hide()

        self.watermarkPanelImageOpacitySlider = QSlider(Qt.Horizontal, self)
        self.watermarkPanelImageOpacitySlider.setFocusPolicy(Qt.NoFocus)
        self.watermarkPanelImageOpacitySlider.setTickPosition(QSlider.NoTicks)
        self.watermarkPanelImageOpacitySlider.setTickInterval(10)
        self.watermarkPanelImageOpacitySlider.setSingleStep(1)
        self.watermarkPanelImageOpacitySlider.setValue(50)
        self.watermarkPanelImageOpacitySlider.setGeometry(self.rx1 + 10 + 55,
                                                          self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 50,
                                                          100, 20)
        self.watermarkPanelImageOpacitySlider.valueChanged.connect(lambda: self.setOpacitySliderValue(1))
        self.watermarkPanelImageOpacitySlider.hide()

        self.watermarkPanelImageOpacitySliderValueLabel = QLabel(self)
        self.watermarkPanelImageOpacitySliderValueLabel.setGeometry(self.rx1 + 10 + 145,
                                                                    self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 30 + 20,
                                                                    50, self.ry2)
        self.watermarkPanelImageOpacitySliderValueLabel.setText("50")
        self.watermarkPanelImageOpacitySliderValueLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.watermarkPanelImageOpacitySliderValueLabel.hide()

        self.watermarkPanelImageOKButton = QPushButton("OK", self)
        self.watermarkPanelImageOKButton.setGeometry(self.rx1 + 10 + 15,
                                                     self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 30 + 20 + 40,
                                                     70, self.ry2 + 5)
        self.watermarkPanelImageOKButton.setIcon(QIcon("img/ok.png"))
        self.watermarkPanelImageOKButton.clicked.connect(self.validateWatermarkImageParameters)
        self.watermarkPanelImageOKButton.hide()

        self.watermarkPanelImageCloseButton = QPushButton("Close", self)
        self.watermarkPanelImageCloseButton.setGeometry(self.rx1 + 10 + 92,
                                                        self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 40 + 30 + 30 + 20 + 40,
                                                        70, self.ry2 + 5)
        self.watermarkPanelImageCloseButton.setIcon(QIcon("img/cancel.png"))
        self.watermarkPanelImageCloseButton.clicked.connect(self.hideWatermarkPanel)
        self.watermarkPanelImageCloseButton.hide()

        # textAddingPanel

        self.textPanelTitle = QLabel(self)
        self.textPanelTitle.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2, self.rx2, self.ry2 + 5)
        self.textPanelTitle.setText("Add Text")
        self.textPanelTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.textPanelTitle.hide()

        self.textPanelTextLabel = QLabel(self)
        self.textPanelTextLabel.setGeometry(self.rx1 + 10, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15, self.rx2,
                                            self.ry2 + 5)
        self.textPanelTextLabel.setText("Text : ")
        self.textPanelTextLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.textPanelTextLabel.hide()

        self.textPanelTextArea = QLineEdit("", self)
        self.textPanelTextArea.setGeometry(self.rx1 + 5 + 40, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 12, 145,
                                           self.ry2)
        self.textPanelTextArea.textChanged.connect(self.updateTextPanelParameters)
        self.textPanelTextArea.hide()

        self.textPanelFontLabel = QLabel(self)
        self.textPanelFontLabel.setGeometry(self.rx1 + 10, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 30,
                                            self.rx2, self.ry2 + 5)
        self.textPanelFontLabel.setText("Font : ")
        self.textPanelFontLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.textPanelFontLabel.hide()

        self.textPanelFontButton = QPushButton("Select", self)
        self.textPanelFontButton.setGeometry(self.rx1 + 10 + 35, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 25,
                                             110, self.ry2 + 5)
        self.textPanelFontButton.setIcon(QIcon("img/choose.png"))
        self.textPanelFontButton.clicked.connect(self.showFontSelectorDialog)
        self.textPanelFontButton.hide()

        self.textPanelFontSizeLabel = QLabel(self)
        self.textPanelFontSizeLabel.setGeometry(self.rx1 + 10 + 35 + 117,
                                                self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 26, 30, self.ry2 + 2)
        self.textPanelFontSizeLabel.setText("0")
        self.textPanelFontSizeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.textPanelFontSizeLabel.setStyleSheet("background-color:white")
        self.textPanelFontSizeLabel.hide()

        self.textPanelColorLabel = QLabel(self)
        self.textPanelColorLabel.setGeometry(self.rx1 + 10, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 60,
                                             self.rx2, self.ry2 + 5)
        self.textPanelColorLabel.setText("Color : ")
        self.textPanelColorLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.textPanelColorLabel.hide()

        self.textPanelColorButton = QPushButton("Pick", self)
        self.textPanelColorButton.setGeometry(self.rx1 + 10 + 35,
                                              self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 25 + 15 + 15, 100,
                                              self.ry2 + 5)
        self.textPanelColorButton.setIcon(QIcon("img/pick.png"))
        self.textPanelColorButton.clicked.connect(self.showColorSelectorDialog)
        self.textPanelColorButton.hide()

        self.textPanelTextCoordinatesLabel = QLabel(self)
        self.textPanelTextCoordinatesLabel.setGeometry(self.rx1 + 10,
                                                       self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 25 + 15 + 50,
                                                       self.rx2, self.ry2 + 5)
        self.textPanelTextCoordinatesLabel.setText("(X,Y) : ")
        self.textPanelTextCoordinatesLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.textPanelTextCoordinatesLabel.hide()

        self.textPanelXCoordinate = QLineEdit("", self)
        self.textPanelXCoordinate.setGeometry(self.rx1 + 10 + 35,
                                              self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 25 + 15 + 48, 50,
                                              self.ry2)
        self.textPanelXCoordinate.textChanged.connect(self.updateTextPanelParameters)
        self.textPanelXCoordinate.hide()

        self.textPanelYCoordinate = QLineEdit("", self)
        self.textPanelYCoordinate.setGeometry(self.rx1 + 10 + 35 + 55,
                                              self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 25 + 15 + 48, 50,
                                              self.ry2)
        self.textPanelYCoordinate.textChanged.connect(self.updateTextPanelParameters)
        self.textPanelYCoordinate.hide()

        self.textPanelOKButton = QPushButton("OK", self)
        self.textPanelOKButton.setGeometry(self.rx1 + 10 + 15,
                                           self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 25 + 15 + 85, 70,
                                           self.ry2 + 5)
        self.textPanelOKButton.setIcon(QIcon("img/ok.png"))
        self.textPanelOKButton.clicked.connect(self.applyText)
        self.textPanelOKButton.hide()

        self.textPanelCancelButton = QPushButton("Close", self)
        self.textPanelCancelButton.setGeometry(self.rx1 + 10 + 92,
                                               self.ry1 + 18 + 18 + 18 + 18 + 22 + 2 + 18 + 15 + 25 + 15 + 85, 70,
                                               self.ry2 + 5)
        self.textPanelCancelButton.setIcon(QIcon("img/cancel.png"))
        self.textPanelCancelButton.clicked.connect(self.hideTextPanel)
        self.textPanelCancelButton.hide()

        # cropPanel

        self.cropPanelTitle = QLabel(self)
        self.cropPanelTitle.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2, self.rx2, self.ry2 + 5)
        self.cropPanelTitle.setText("Crop Image")
        self.cropPanelTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.cropPanelTitle.hide()

        self.cropPanelXLabel = QLabel(self)
        self.cropPanelXLabel.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15, 25, self.ry2)
        self.cropPanelXLabel.setText("X:")
        self.cropPanelXLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.cropPanelXLabel.hide()

        self.cropPanelXTextArea = QLineEdit("0", self)
        self.cropPanelXTextArea.setGeometry(self.rx1 + 45, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15, 50, self.ry2)
        self.cropPanelXTextArea.textChanged.connect(self.updateCropParameters)
        self.cropPanelXTextArea.hide()

        self.cropPanelYLabel = QLabel(self)
        self.cropPanelYLabel.setGeometry(self.rx1 + 26 + 50 + 23, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15, 25,
                                         self.ry2)
        self.cropPanelYLabel.setText("Y:")
        self.cropPanelYLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.cropPanelYLabel.hide()

        self.cropPanelYTextArea = QLineEdit("0", self)
        self.cropPanelYTextArea.setGeometry(self.rx1 + 26 + 50 + 25 + 45, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15,
                                            50, self.ry2)
        self.cropPanelYTextArea.textChanged.connect(self.updateCropParameters)
        self.cropPanelYTextArea.hide()

        self.cropPanelHeightLabel = QLabel(self)
        self.cropPanelHeightLabel.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 45, self.ry2)
        self.cropPanelHeightLabel.setText("Height:")
        self.cropPanelHeightLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.cropPanelHeightLabel.hide()

        self.cropPanelHeightTextArea = QLineEdit("0", self)
        self.cropPanelHeightTextArea.setGeometry(self.rx1 + 45, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 50,
                                                 self.ry2)
        self.cropPanelHeightTextArea.textChanged.connect(self.updateCropParameters)
        self.cropPanelHeightTextArea.hide()

        self.cropPanelWidthLabel = QLabel(self)
        self.cropPanelWidthLabel.setGeometry(self.rx1 + 48 + 53, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 45,
                                             self.ry2)
        self.cropPanelWidthLabel.setText("Width:")
        self.cropPanelWidthLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.cropPanelWidthLabel.hide()

        self.cropPanelWidthTextArea = QLineEdit("0", self)
        self.cropPanelWidthTextArea.setGeometry(self.rx1 + 48 + 53 + 45,
                                                self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 50, self.ry2)
        self.cropPanelWidthTextArea.textChanged.connect(self.updateCropParameters)
        self.cropPanelWidthTextArea.hide()

        self.cropPanelPreviewButton = QPushButton("Preview", self)
        self.cropPanelPreviewButton.setGeometry(self.rx1 + 20 + 15,
                                                self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20 + 20 + 5, 65,
                                                self.ry2 + 10)
        self.cropPanelPreviewButton.setIcon(QIcon("img/previewButton.png"))
        self.cropPanelPreviewButton.clicked.connect(self.getPreviewForCrop)
        self.cropPanelPreviewButton.hide()

        self.cropPanelCropButton = QPushButton("Crop", self)
        self.cropPanelCropButton.setGeometry(self.rx1 + 20 + 50 + 10 + 25,
                                             self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20 + 20 + 5, 50,
                                             self.ry2 + 10)
        self.cropPanelCropButton.setIcon(QIcon("img/cropButton.png"))
        self.cropPanelCropButton.clicked.connect(self.performCrop)
        self.cropPanelCropButton.hide()

        # flipPanel

        self.flipPanelTitle = QLabel(self)
        self.flipPanelTitle.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2, self.rx2, self.ry2 + 5)
        self.flipPanelTitle.setText("Flip Image")
        self.flipPanelTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.flipPanelTitle.hide()

        self.flipPanelHorizontalRadioButton = QRadioButton(self)
        self.flipPanelHorizontalRadioButton.setGeometry(self.rx1 + 20, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15, 80,
                                                        self.ry2)
        self.flipPanelHorizontalRadioButton.setText("Horizontal")
        self.flipPanelHorizontalRadioButton.hide()

        self.flipPanelVerticalRadioButton = QRadioButton(self)
        self.flipPanelVerticalRadioButton.setGeometry(self.rx1 + 120, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15, 80,
                                                      self.ry2)
        self.flipPanelVerticalRadioButton.setText("Vertical")
        self.flipPanelVerticalRadioButton.hide()

        self.flipPanelOKButton = QPushButton("OK", self)
        self.flipPanelOKButton.setGeometry(self.rx1 + 20 + 10, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 23, 70,
                                           self.ry2 + 10)
        self.flipPanelOKButton.setIcon(QIcon("img/ok.png"))
        self.flipPanelOKButton.clicked.connect(self.performFlip)
        self.flipPanelOKButton.hide()

        self.flipPanelCancelButton = QPushButton("Close", self)
        self.flipPanelCancelButton.setGeometry(self.rx1 + 20 + 15 + 70,
                                               self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 23, 70, self.ry2 + 10)
        self.flipPanelCancelButton.setIcon(QIcon("img/cancel.png"))
        self.flipPanelCancelButton.clicked.connect(self.hideFlipPanel)
        self.flipPanelCancelButton.hide()

        # rotatePanel

        self.rotatePanelTitle = QLabel(self)
        self.rotatePanelTitle.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2, self.rx2, self.ry2 + 5)
        self.rotatePanelTitle.setText("Rotate Image")
        self.rotatePanelTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.rotatePanelTitle.hide()

        self.rotatePanelClockwiseRadioButton = QRadioButton(self)
        self.rotatePanelClockwiseRadioButton.setGeometry(self.rx1 + 5, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15, 110,
                                                         self.ry2)
        self.rotatePanelClockwiseRadioButton.setText("Clockwise - 90°")
        self.rotatePanelClockwiseRadioButton.hide()

        self.rotatePanelAntiClockwiseRadioButton = QRadioButton(self)
        self.rotatePanelAntiClockwiseRadioButton.setGeometry(self.rx1 + 5,
                                                             self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15 + 30, 110,
                                                             self.ry2)
        self.rotatePanelAntiClockwiseRadioButton.setText("Anticlockwise - 90°")
        self.rotatePanelAntiClockwiseRadioButton.hide()

        self.rotatePanelOneEightyDegreeRadioButton = QRadioButton(self)
        self.rotatePanelOneEightyDegreeRadioButton.setGeometry(self.rx1 + 5,
                                                               self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15 + 30 + 30,
                                                               110, self.ry2)
        self.rotatePanelOneEightyDegreeRadioButton.setText("180°")
        self.rotatePanelOneEightyDegreeRadioButton.hide()

        self.rotatePanelOKButton = QPushButton("OK", self)
        self.rotatePanelOKButton.setGeometry(self.rx1 + 20 + 10,
                                             self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15 + 30 + 30 + 23 + 5, 70,
                                             self.ry2 + 10)
        self.rotatePanelOKButton.setIcon(QIcon("img/ok.png"))
        self.rotatePanelOKButton.clicked.connect(self.performRotation)
        self.rotatePanelOKButton.hide()

        self.rotatePanelCancelButton = QPushButton("Close", self)
        self.rotatePanelCancelButton.setGeometry(self.rx1 + 20 + 15 + 70,
                                                 self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 15 + 30 + 30 + 23 + 5, 70,
                                                 self.ry2 + 10)
        self.rotatePanelCancelButton.setIcon(QIcon("img/cancel.png"))
        self.rotatePanelCancelButton.clicked.connect(self.hideRotatePanel)
        self.rotatePanelCancelButton.hide()

        # overlayPanel

        self.overlayPanelTitle = QLabel(self)
        self.overlayPanelTitle.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2, self.rx2, self.ry2 + 5)
        self.overlayPanelTitle.setText("Overlay")
        self.overlayPanelTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.overlayPanelTitle.hide()

        self.overlayPanelImageTitle = QLabel(self)
        self.overlayPanelImageTitle.setGeometry(self.rx1 + 10, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 10, 45,
                                                self.ry2)
        self.overlayPanelImageTitle.setText("Image : ")
        self.overlayPanelImageTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.overlayPanelImageTitle.hide()

        self.overlayPanelImageButton = QPushButton("Choose...", self)
        self.overlayPanelImageButton.setGeometry(self.rx1 + 10 + 45, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 10, 95,
                                                 self.ry2 + 5)
        self.overlayPanelImageButton.setIcon(QIcon("img/choose.png"))
        self.overlayPanelImageButton.clicked.connect(self.openOverlayImageChooser)
        self.overlayPanelImageButton.hide()

        self.overlayPanelImageLabel = QLabel(self)
        self.overlayPanelImageLabel.setGeometry(self.rx1 + 10 + 30, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 8, 110,
                                                self.ry2 + 5)
        self.overlayPanelImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.overlayPanelImageLabel.hide()

        self.overlayPanelXLabel = QLabel(self)
        self.overlayPanelXLabel.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 45, self.ry2)
        self.overlayPanelXLabel.setText("X:")
        self.overlayPanelXLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.overlayPanelXLabel.hide()

        self.overlayPanelXTextArea = QLineEdit("0", self)
        self.overlayPanelXTextArea.setGeometry(self.rx1 + 45, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 50,
                                               self.ry2)
        self.overlayPanelXTextArea.textChanged.connect(self.updateOverlayParameters)
        self.overlayPanelXTextArea.hide()

        self.overlayPanelYLabel = QLabel(self)
        self.overlayPanelYLabel.setGeometry(self.rx1 + 48 + 53, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 45,
                                            self.ry2)
        self.overlayPanelYLabel.setText("Y:")
        self.overlayPanelYLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.overlayPanelYLabel.hide()

        self.overlayPanelYTextArea = QLineEdit("0", self)
        self.overlayPanelYTextArea.setGeometry(self.rx1 + 48 + 53 + 45,
                                               self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 50, self.ry2)
        self.overlayPanelYTextArea.textChanged.connect(self.updateOverlayParameters)
        self.overlayPanelYTextArea.hide()

        self.overlayPanelOKButton = QPushButton("OK", self)
        self.overlayPanelOKButton.setGeometry(self.rx1 + 20 + 10,
                                              self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20 + 20 + 10, 70,
                                              self.ry2 + 10)
        self.overlayPanelOKButton.setIcon(QIcon("img/ok.png"))
        self.overlayPanelOKButton.clicked.connect(self.performOverlay)
        self.overlayPanelOKButton.hide()

        self.overlayPanelCancelButton = QPushButton("Close", self)
        self.overlayPanelCancelButton.setGeometry(self.rx1 + 20 + 15 + 70,
                                                  self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20 + 20 + 10, 70,
                                                  self.ry2 + 10)
        self.overlayPanelCancelButton.setIcon(QIcon("img/cancel.png"))
        self.overlayPanelCancelButton.clicked.connect(self.hideOverlayPanel)
        self.overlayPanelCancelButton.hide()

        # Canny Threshold Panel
        self.cannyPanelTitle = QLabel(self)
        self.cannyPanelTitle.setGeometry(self.rx1, self.ry1 + 18 + 18 + 18 + 18 + 22 + 2, self.rx2, self.ry2 + 5)
        self.cannyPanelTitle.setText("Canny Threshold")
        self.cannyPanelTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.cannyPanelTitle.hide()


        self.cannyPanelThreshold1Label = QLabel(self)
        self.cannyPanelThreshold1Label.setGeometry(self.rx1 + 30, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 45 + 50, self.ry2)
        self.cannyPanelThreshold1Label.setText("Threshold 1:")
        self.cannyPanelThreshold1Label.setAlignment(QtCore.Qt.AlignCenter)
        self.cannyPanelThreshold1Label.hide()

        self.cannyPanelThreshold1Area = QLineEdit("0", self)
        self.cannyPanelThreshold1Area.setGeometry(self.rx1 + 45 + 50 + 40, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20, 50,
                                               self.ry2)
        self.cannyPanelThreshold1Area.textChanged.connect(self.updateCannyParameters)
        self.cannyPanelThreshold1Area.hide()

        self.cannyPanelThreshold2Label = QLabel(self)
        self.cannyPanelThreshold2Label.setGeometry(self.rx1 + 28 + 3, self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20 + 20, 45 + 50,
                                            self.ry2)
        self.cannyPanelThreshold2Label.setText("Threshold 2:")
        self.cannyPanelThreshold2Label.setAlignment(QtCore.Qt.AlignCenter)
        self.cannyPanelThreshold2Label.hide()

        self.cannyPanelThreshold2Area = QLineEdit("0", self)
        self.cannyPanelThreshold2Area.setGeometry(self.rx1 + 48 + 2 + 45 + 40,
                                               self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20 + 20, 50, self.ry2)
        self.cannyPanelThreshold2Area.textChanged.connect(self.updateCannyParameters)
        self.cannyPanelThreshold2Area.hide()

        self.cannyPanelOKButton = QPushButton("OK", self)
        self.cannyPanelOKButton.setGeometry(self.rx1 + 20 + 10,
                                              self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20 + 20 + 10 + 20, 70,
                                              self.ry2 + 10)
        self.cannyPanelOKButton.setIcon(QIcon("img/ok.png"))
        self.cannyPanelOKButton.clicked.connect(self.performCannyEdgeDetection)
        self.cannyPanelOKButton.hide()

        self.cannyPanelCancelButton = QPushButton("Close", self)
        self.cannyPanelCancelButton.setGeometry(self.rx1 + 20 + 15 + 70,
                                                  self.ry1 + 18 + 18 + 18 + 18 + 24 + 18 + 20 + 20 + 20 + 10 + 20, 70,
                                                  self.ry2 + 10)
        self.cannyPanelCancelButton.setIcon(QIcon("img/cancel.png"))
        self.cannyPanelCancelButton.clicked.connect(self.hideCannyPanel)
        self.cannyPanelCancelButton.hide()


    # 更新Bars状态
    def updateBars(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.toolbar.setEnabled(True)
        self.saveAct.setEnabled(True)
        self.brightnessAct.setEnabled(True)
        self.contrastAct.setEnabled(True)
        self.resizeAct.setEnabled(True)
        self.zoomInAct.setEnabled(True)
        self.zoomOutAct.setEnabled(True)
        self.originalAct.setEnabled(True)
        # self.blurMenu.setEnabled(True)
        # self.borderAct.setEnabled(True)
        self.change1Act.setEnabled(True)
        self.change2Act.setEnabled(True)
        self.change3Act.setEnabled(True)
        self.noise1Act.setEnabled(True)
        self.noise2Act.setEnabled(True)
        self.smoothing1Act.setEnabled(True)
        self.smoothing2Act.setEnabled(True)
        self.smoothing3Act.setEnabled(True)
        self.smoothing4Act.setEnabled(True)
        self.smoothing5Act.setEnabled(True)
        self.hist1Act.setEnabled(True)
        self.hist2Act.setEnabled(True)
        self.enhance1Act.setEnabled(True)
        self.enhance2Act.setEnabled(True)
        self.enhance3Act.setEnabled(True)
        self.enhance4Act.setEnabled(True)
        self.enhance5Act.setEnabled(True)
        self.enhance6Act.setEnabled(True)
        self.segment1Act.setEnabled(True)
        self.segment3Act.setEnabled(True)
        self.morphologyProcess1Act.setEnabled(True)
        self.morphologyProcess2Act.setEnabled(True)
        self.cornerDetectionAct.setEnabled(True)
        self.cannyEdgeDetectionAct.setEnabled(True)
        self.imgButton.setEnabled(True)

    # 显示图片&更新Bar状态
    def update(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.updateImageOnUI()
        self.updateBars()

    def showSliderValue(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        print(self.slider.value())

    # 创建StatusBar 显示信息
    def createStatusBar(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet("background:#FFFCE3")
        self.statusBar.setFixedHeight(30)
        self.setStatusBar(self.statusBar)
        self.images["status"] = "Select an Image..."
        self.updateStatusBar()

    def updateResizeParameters(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        try:
            self.destHeight = int(self.resizeHeightTextArea.text())
            self.destWidth = int(self.resizeWidthTextArea.text())
        except:
            self.destHeight = 0
            self.destWidth = 0

    def updateOverlayParameters(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.startingOverlayX = self.overlayPanelXTextArea.text()
        self.startingOverlayY = self.overlayPanelYTextArea.text()

    def updateCannyParameters(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.threshold1 = self.cannyPanelThreshold1Area.text()
        self.threshold2 = self.cannyPanelThreshold2Area.text()
        print(f'threshold1 {self.threshold1}')
        print(f'threshold2 {self.threshold2}')

    def updateCropParameters(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.startingPreviewX = self.cropPanelXTextArea.text()
        self.startingPreviewY = self.cropPanelYTextArea.text()
        self.endingPreviewX = self.cropPanelHeightTextArea.text()
        self.endingPreviewY = self.cropPanelWidthTextArea.text()

    def getPreviewForCrop(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(4)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performOverlay(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(11)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performFlip(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if self.flipPanelHorizontalRadioButton.isChecked():
            self.worker.setTmApplication(self)
            self.worker.setOperation(6)
            self.worker.signal.connect(self.taskFinished)
            self.worker.start()
            self.showLoader()
            self.hideCropPanel()
        elif self.flipPanelVerticalRadioButton.isChecked():
            self.worker.setTmApplication(self)
            self.worker.setOperation(7)
            self.worker.signal.connect(self.taskFinished)
            self.worker.start()
            self.showLoader()
            self.hideCropPanel()

    def performCrop(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(5)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()
        self.hideCropPanel()

    def performRotation(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if self.rotatePanelClockwiseRadioButton.isChecked():
            self.worker.setTmApplication(self)
            self.worker.setOperation(8)
            self.worker.signal.connect(self.taskFinished)
            self.worker.start()
            self.showLoader()
            self.hideCropPanel()
        elif self.rotatePanelAntiClockwiseRadioButton.isChecked():
            self.worker.setTmApplication(self)
            self.worker.setOperation(9)
            self.worker.signal.connect(self.taskFinished)
            self.worker.start()
            self.showLoader()
            self.hideCropPanel()
        elif self.rotatePanelOneEightyDegreeRadioButton.isChecked():
            self.worker.setTmApplication(self)
            self.worker.setOperation(10)
            self.worker.signal.connect(self.taskFinished)
            self.worker.start()
            self.showLoader()
            self.hideCropPanel()

    def hideRotatePanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.rotatePanelTitle.hide()
        self.rotatePanelClockwiseRadioButton.hide()
        self.rotatePanelAntiClockwiseRadioButton.hide()
        self.rotatePanelOneEightyDegreeRadioButton.hide()
        self.rotatePanelOKButton.hide()
        self.rotatePanelCancelButton.hide()

    def hideFlipPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.flipPanelTitle.hide()
        self.flipPanelHorizontalRadioButton.hide()
        self.flipPanelVerticalRadioButton.hide()
        self.flipPanelOKButton.hide()
        self.flipPanelCancelButton.hide()

    def hideOverlayPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.overlayPanelTitle.hide()
        self.overlayPanelImageTitle.hide()
        self.overlayPanelImageButton.hide()
        self.overlayPanelXLabel.hide()
        self.overlayPanelXTextArea.hide()
        self.overlayPanelYLabel.hide()
        self.overlayPanelYTextArea.hide()
        self.overlayPanelOKButton.hide()
        self.overlayPanelCancelButton.hide()
        self.overlayPanelImageLabel.hide()

    def showTextPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.hideCropPanel()
        self.hideOverlayPanel()
        self.hideFlipPanel()
        self.hideRotatePanel()
        self.hideCannyPanel()
        self.textPanelTitle.show()
        self.textPanelTextLabel.show()
        self.textPanelTextArea.show()
        self.textPanelFontLabel.show()
        self.textPanelFontButton.show()
        self.textPanelFontSizeLabel.show()
        self.textPanelColorLabel.show()
        self.textPanelColorButton.show()
        self.textPanelTextCoordinatesLabel.show()
        self.textPanelXCoordinate.show()
        self.textPanelYCoordinate.show()
        self.textPanelOKButton.show()
        self.textPanelCancelButton.show()

    def hideTextPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.textPanelTitle.hide()
        self.textPanelTextLabel.hide()
        self.textPanelTextArea.hide()
        self.textPanelFontLabel.hide()
        self.textPanelFontButton.hide()
        self.textPanelFontSizeLabel.hide()
        self.textPanelColorLabel.hide()
        self.textPanelColorButton.hide()
        self.textPanelTextCoordinatesLabel.hide()
        self.textPanelXCoordinate.hide()
        self.textPanelYCoordinate.hide()
        self.textPanelOKButton.hide()
        self.textPanelCancelButton.hide()

    def showOverlayPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.hideCropPanel()
        self.hideTextPanel()
        self.hideFlipPanel()
        self.hideRotatePanel()
        self.overlayPanelTitle.show()
        self.overlayPanelImageTitle.show()
        self.overlayPanelImageButton.show()
        self.overlayPanelXLabel.show()
        self.overlayPanelXTextArea.show()
        self.overlayPanelYLabel.show()
        self.overlayPanelYTextArea.show()
        self.overlayPanelOKButton.show()
        self.overlayPanelCancelButton.show()

    def showFlipPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.hideCropPanel()
        self.hideOverlayPanel()
        self.hideCannyPanel()
        self.hideTextPanel()
        self.hideRotatePanel()
        self.flipPanelTitle.show()
        self.flipPanelHorizontalRadioButton.show()
        self.flipPanelVerticalRadioButton.show()
        self.flipPanelOKButton.show()
        self.flipPanelCancelButton.show()

    def showRotatePanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.hideCropPanel()
        self.hideOverlayPanel()
        self.hideCannyPanel()
        self.hideFlipPanel()
        self.hideTextPanel()
        self.rotatePanelTitle.show()
        self.rotatePanelClockwiseRadioButton.show()
        self.rotatePanelAntiClockwiseRadioButton.show()
        self.rotatePanelOneEightyDegreeRadioButton.show()
        self.rotatePanelOKButton.show()
        self.rotatePanelCancelButton.show()

    def showCropPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.hideTextPanel()
        self.hideOverlayPanel()
        self.hideCannyPanel()
        self.hideFlipPanel()
        self.hideRotatePanel()
        self.cropPanelTitle.show()
        self.cropPanelXLabel.show()
        self.cropPanelXTextArea.show()
        self.cropPanelYLabel.show()
        self.cropPanelYTextArea.show()
        self.cropPanelHeightLabel.show()
        self.cropPanelHeightTextArea.show()
        self.cropPanelWidthLabel.show()
        self.cropPanelWidthTextArea.show()
        self.cropPanelPreviewButton.show()
        self.cropPanelCropButton.show()

    def hideCropPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.cropPanelTitle.hide()
        self.cropPanelXLabel.hide()
        self.cropPanelXTextArea.hide()
        self.cropPanelYLabel.hide()
        self.cropPanelYTextArea.hide()
        self.cropPanelHeightLabel.hide()
        self.cropPanelHeightTextArea.hide()
        self.cropPanelWidthLabel.hide()
        self.cropPanelWidthTextArea.hide()
        self.cropPanelPreviewButton.hide()
        self.cropPanelCropButton.hide()

    def performGrayScale(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(1)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    # 处理过程中显示加载图片
    def showLoader(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.loader = QMovie("img/loader.gif")
        self.loaderLabel = QLabel(self)
        self.loaderLabel.setGeometry(590, 220, 120, 120)
        # self.loaderLabel.setStyleSheet("border:2px solid black")
        self.loaderLabel.setMovie(self.loader)
        self.loader.start()
        self.loaderLabel.show()

    # 隐藏加载图片&更新
    def hideLoader(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.loaderLabel.setVisible(False)
        self.update()

    def taskFinished(self, result):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.updateStatusBar()
        self.hideLoader()

    # 创建ToolBar  [Gray Scale, Flip, Rotation, Crop, ...]
    def createToolBar(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # grayScaleAct = QAction(QtGui.QIcon('img/gs.png'), 'Gray Scale', self)
        # grayScaleAct.setStatusTip('Gray Scale')
        # grayScaleAct.triggered.connect(self.performGrayScale)

        self.toolbarWidget = QWidget()
        self.toolbarWidget.setFixedHeight(40)
        self.toolbar = self.addToolBar('Tool Bar')
        self.toolbar.addWidget(self.toolbarWidget)

        self.hboxLayout = QHBoxLayout()  # 设置水平布局
        self.toolbarWidget.setLayout(self.hboxLayout)

        self.changeButton = QPushButton('变换', self.toolbarWidget)
        self.changeButton.setStyleSheet("border:none")
        self.changeButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.hboxLayout.addWidget(self.changeButton)
        changemenu = QMenu(self)
        self.change1Act = QAction('傅里叶变换', self)
        self.change1Act.triggered.connect(self.performFourierTransform)
        changemenu.addAction(self.change1Act)
        self.change2Act = QAction('离散余弦变换', self)
        self.change2Act.triggered.connect(self.performDiscreteCosineTransform)
        changemenu.addAction(self.change2Act)
        self.change3Act = QAction('Radon变换', self)
        self.change3Act.triggered.connect(self.performDiscreteRadonTransform)
        changemenu.addAction(self.change3Act)
        self.changeButton.setMenu(changemenu)
        # 噪声按钮
        self.noiseButton = QPushButton('噪声', self.toolbarWidget)
        self.noiseButton.setStyleSheet("border:none")
        self.noiseButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.hboxLayout.addWidget(self.noiseButton)
        noisemenu = QMenu(self)
        self.noise1Act = QAction('高斯噪声', self)
        self.noise1Act.triggered.connect(self.performGaussianNoiseImage)
        noisemenu.addAction(self.noise1Act)
        self.noise2Act = QAction('椒盐噪声', self)
        self.noise2Act.triggered.connect(self.performSaltPepperImage)
        noisemenu.addAction(self.noise2Act)
        # self.noise3Act = QAction('斑点噪声', self)
        # noisemenu.addAction(self.noise3Act)
        # self.noise4Act = QAction('泊松噪声', self)
        # noisemenu.addAction(self.noise4Act)
        self.noiseButton.setMenu(noisemenu)
        # 滤波按钮
        self.smoothingButton = QPushButton('滤波', self.toolbarWidget)
        self.smoothingButton.setStyleSheet("border:none")
        self.smoothingButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.hboxLayout.addWidget(self.smoothingButton)
        smoothingmenu = QMenu(self)
        self.smoothing1Act = QAction('高通滤波', self)
        self.smoothing1Act.triggered.connect(self.performHighPassFilter)
        smoothingmenu.addAction(self.smoothing1Act)
        self.smoothing2Act = QAction('低通滤波', self)
        self.smoothing2Act.triggered.connect(self.performMedianBlur)
        smoothingmenu.addAction(self.smoothing2Act)
        self.smoothing3Act = QAction('平滑滤波', self)
        self.smoothing3Act.triggered.connect(self.performBlur)
        smoothingmenu.addAction(self.smoothing3Act)
        self.smoothing4Act = QAction('锐化滤波', self)
        # self.smoothing4Act.setEnabled(False)
        self.smoothing4Act.triggered.connect(self.performBilateralFilter)
        smoothingmenu.addAction(self.smoothing4Act)
        self.smoothing5Act = QAction('高斯滤波', self)
        # self.smoothing5Act.setEnabled(False)
        self.smoothing5Act.triggered.connect(self.createGuassianBlurWindow)
        smoothingmenu.addAction(self.smoothing5Act)
        self.smoothingButton.setMenu(smoothingmenu)
        # 直方图统计按钮
        self.histButton = QPushButton('直方图统计', self.toolbarWidget)
        self.histButton.setStyleSheet("border:none")
        self.histButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.hboxLayout.addWidget(self.histButton)
        histmenu = QMenu(self)
        self.hist1Act = QAction('颜色直方图', self)
        self.hist1Act.triggered.connect(self.performPixelFrequencyHistogram)
        histmenu.addAction(self.hist1Act)
        self.hist2Act = QAction('灰度直方图', self)
        self.hist2Act.triggered.connect(self.performGrayFrequencyHistogram)
        histmenu.addAction(self.hist2Act)
        # self.hist3Act = QAction('B直方图', self)
        # histmenu.addAction(self.hist3Act)
        self.histButton.setMenu(histmenu)
        # 图像增强按钮
        self.enhanceButton = QPushButton('图像增强', self.toolbarWidget)
        self.enhanceButton.setStyleSheet("border:none")
        self.enhanceButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.hboxLayout.addWidget(self.enhanceButton)
        enhancemenu = QMenu(self)
        self.enhance1Act = QAction('伪彩色增强', self)
        self.enhance1Act.triggered.connect(self.performpSeudoColorEnhancement)
        enhancemenu.addAction(self.enhance1Act)
        self.enhance2Act = QAction('真彩色增强', self)
        self.enhance2Act.triggered.connect(self.performTrueColorEnhancement)
        enhancemenu.addAction(self.enhance2Act)
        self.enhance3Act = QAction('直方图均衡', self)
        self.enhance3Act.triggered.connect(self.performHistNormalized)
        enhancemenu.addAction(self.enhance3Act)
        self.enhance4Act = QAction('NTSC颜色模型', self)
        self.enhance4Act.triggered.connect(self.performBGR2NTSC)
        enhancemenu.addAction(self.enhance4Act)
        self.enhance5Act = QAction('YCbCr颜色模型', self)
        self.enhance5Act.triggered.connect(self.performBGR2YCR_CB)
        enhancemenu.addAction(self.enhance5Act)
        self.enhance6Act = QAction('HSV颜色模型', self)
        self.enhance6Act.triggered.connect(self.performBGR2HSV)
        enhancemenu.addAction(self.enhance6Act)
        self.enhanceButton.setMenu(enhancemenu)
        # 图像分割按钮
        self.segmentButton = QPushButton('图像分割', self.toolbarWidget)
        self.segmentButton.setStyleSheet("border:none")
        self.segmentButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.hboxLayout.addWidget(self.segmentButton)
        segmentmenu = QMenu(self)
        self.segment1Act = QAction('二值阈值分割', self)
        self.segment1Act.triggered.connect(self.performBinalization)
        segmentmenu.addAction(self.segment1Act)
        self.segment3Act = QAction('Otsu阈值分割', self)
        self.segment3Act.triggered.connect(self.performOtsuBinarization)
        segmentmenu.addAction(self.segment3Act)
        # self.segment2Act = QAction('canny边缘检测', self)
        # self.segment2Act.triggered.connect(self.performCannyEdgeDetection)
        # segmentmenu.addAction(self.segment2Act)
        # self.segment3Act = QAction('霍夫变换', self)
        # segmentmenu.addAction(self.segment3Act)
        self.segmentButton.setMenu(segmentmenu)
        # 形态学处理按钮
        self.morphologyProcessButton = QPushButton('形态学处理', self.toolbarWidget)
        self.morphologyProcessButton.setStyleSheet("border:none")
        self.morphologyProcessButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.hboxLayout.addWidget(self.morphologyProcessButton)
        morphologyProcessmenu = QMenu(self)
        self.morphologyProcess1Act = QAction('膨胀', self)
        self.morphologyProcess1Act.triggered.connect(self.performDilateImage)
        morphologyProcessmenu.addAction(self.morphologyProcess1Act)
        self.morphologyProcess2Act = QAction('腐蚀', self)
        self.morphologyProcess2Act.triggered.connect(self.performErodeImage)
        morphologyProcessmenu.addAction(self.morphologyProcess2Act)
        self.morphologyProcessButton.setMenu(morphologyProcessmenu)
        # 特征提取按钮
        self.featureExtractionProcessButton = QPushButton('特征提取', self.toolbarWidget)
        self.featureExtractionProcessButton.setStyleSheet("border:none")
        self.featureExtractionProcessButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.hboxLayout.addWidget(self.featureExtractionProcessButton)
        featureExtractionmenu = QMenu(self)
        self.cornerDetectionAct = QAction('角点检测', self)
        self.cornerDetectionAct.triggered.connect(self.performFeatureExtraction)
        featureExtractionmenu.addAction(self.cornerDetectionAct)
        self.cannyEdgeDetectionAct = QAction('canny边缘检测', self)
        self.cannyEdgeDetectionAct.triggered.connect(self.showCannyPanel)
        featureExtractionmenu.addAction(self.cannyEdgeDetectionAct)
        self.featureExtractionProcessButton.setMenu(featureExtractionmenu)
        # 图像分类与识别按钮
        self.imgButton = QPushButton('图像分类与识别', self.toolbarWidget)
        self.imgButton.setStyleSheet("border:none")
        self.imgButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.imgButton.clicked.connect(self.performClassification)
        self.hboxLayout.addWidget(self.imgButton)

        self.change1Act.setEnabled(False)
        self.change2Act.setEnabled(False)
        self.change3Act.setEnabled(False)
        self.noise1Act.setEnabled(False)
        self.noise2Act.setEnabled(False)
        self.smoothing1Act.setEnabled(False)
        self.smoothing2Act.setEnabled(False)
        self.smoothing3Act.setEnabled(False)
        self.smoothing4Act.setEnabled(False)
        self.smoothing5Act.setEnabled(False)
        self.hist1Act.setEnabled(False)
        self.hist2Act.setEnabled(False)
        self.enhance1Act.setEnabled(False)
        self.enhance2Act.setEnabled(False)
        self.enhance3Act.setEnabled(False)
        self.enhance4Act.setEnabled(False)
        self.enhance5Act.setEnabled(False)
        self.enhance6Act.setEnabled(False)
        self.segment1Act.setEnabled(False)
        self.segment3Act.setEnabled(False)
        self.morphologyProcess1Act.setEnabled(False)
        self.morphologyProcess2Act.setEnabled(False)
        self.cornerDetectionAct.setEnabled(False)
        self.cannyEdgeDetectionAct.setEnabled(False)
        self.imgButton.setEnabled(False)

        # toolbar settings
        self.toolbar.setMovable(False)
        self.toolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        # self.toolbar.layout().setSpacing(7)
        # self.toolbar.layout().setContentsMargins(6, 15, 4, 5)
        self.toolbar.layout().setContentsMargins(0, 0, 0, 0)
        self.toolbar.setFixedHeight(40)
        self.toolbar.setStyleSheet("background:#FFFCE3")  # 背景色
        self.toolbar.setEnabled(True)

    # 创建MenuBar [File, Edit, View, Transform]
    def createMenuBar(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        menuBar = QMenuBar(self)
        # file
        fileMenu = QMenu("&File", self)

        openAct = QAction(QtGui.QIcon('img/open.png'), '&Open', self)
        openAct.setShortcut('Ctrl+O')
        openAct.setStatusTip('Open File')
        openAct.triggered.connect(self.openFileChooser)
        fileMenu.addAction(openAct)

        self.saveAct = QAction(QtGui.QIcon('img/save.png'), '&Save', self)
        self.saveAct.setShortcut('Ctrl+S')
        self.saveAct.setStatusTip('Save Image')
        self.saveAct.triggered.connect(self.saveFileChooser)
        self.saveAct.setEnabled(False)
        fileMenu.addAction(self.saveAct)

        exitAct = QAction(QtGui.QIcon('img/exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(QtWidgets.qApp.quit)
        fileMenu.addAction(exitAct)

        # edit
        editMenu = QMenu("&Edit", self)

        self.undoAct = QAction(QtGui.QIcon('img/undo.png'), '&Undo...', self)
        self.undoAct.setShortcut('Ctrl+Z')
        self.undoAct.setStatusTip('Undo...')
        self.undoAct.triggered.connect(self.performUndo)
        self.undoAct.setEnabled(False)
        editMenu.addAction(self.undoAct)

        self.redoAct = QAction(QtGui.QIcon('img/redo.png'), '&Redo...', self)
        self.redoAct.setShortcut('Ctrl+Y')
        self.redoAct.setStatusTip('Redo...')
        self.redoAct.triggered.connect(self.performRedo)
        self.redoAct.setEnabled(False)
        editMenu.addAction(self.redoAct)
        editMenu.addSeparator()

        self.brightnessAct = QAction(QtGui.QIcon('img/brightness.png'), '&Brightness', self)
        self.brightnessAct.setShortcut('Ctrl+B')
        self.brightnessAct.setStatusTip('Set Brightness')
        self.brightnessAct.triggered.connect(self.createBrightnessWindow)
        self.brightnessAct.setEnabled(False)
        editMenu.addAction(self.brightnessAct)

        self.contrastAct = QAction(QtGui.QIcon('img/contrast.png'), '&Contrast', self)
        self.contrastAct.setShortcut('Ctrl+C')
        self.contrastAct.setStatusTip('Set Contrast')
        self.contrastAct.triggered.connect(self.createContrastWindow)
        self.contrastAct.setEnabled(False)
        editMenu.addAction(self.contrastAct)
        editMenu.addSeparator()

        self.resizeAct = QAction(QtGui.QIcon('img/resize.png'), '&Resize', self)
        self.resizeAct.setShortcut('Ctrl+R')
        self.resizeAct.setStatusTip('Resize image')
        self.resizeAct.triggered.connect(self.createResizeWindow)
        self.resizeAct.setEnabled(False)
        editMenu.addAction(self.resizeAct)

        # View
        viewMenu = QMenu("&View", self)

        self.zoomInAct = QAction(QtGui.QIcon('img/zoom-in.png'), '&Zoom In', self)
        self.zoomInAct.setStatusTip('Zoom In')
        self.zoomInAct.setShortcut('Ctrl+K')
        self.zoomInAct.triggered.connect(self.performZoomIn)
        self.zoomInAct.setEnabled(False)
        viewMenu.addAction(self.zoomInAct)

        self.zoomOutAct = QAction(QtGui.QIcon('img/zoom-out.png'), '&Zoom Out', self)
        self.zoomOutAct.setStatusTip('Zoom Out')
        self.zoomOutAct.setShortcut('Ctrl+L')
        self.zoomOutAct.triggered.connect(self.performZoomOut)
        self.zoomOutAct.setEnabled(False)
        viewMenu.addAction(self.zoomOutAct)
        viewMenu.addSeparator()

        # # transform
        # transformMenu = QMenu("&Transform", self)
        # self.blurMenu = QMenu('&Smoothing', self)
        # self.blurMenu.setStatusTip('Smoothing')
        # self.blurMenu.setEnabled(False)
        # transformMenu.addMenu(self.blurMenu)
        #
        # self.gaussianBlurTransformAct = QAction(QtGui.QIcon('img/guassianBlur.png'), '&Guassian Blur', self)
        # self.gaussianBlurTransformAct.setStatusTip('Guassian Blur')
        # self.gaussianBlurTransformAct.triggered.connect(self.createGuassianBlurWindow)
        # # self.gaussianSmoothingTransformAct.setEnabled(False)
        # self.blurMenu.addAction(self.gaussianBlurTransformAct)
        #
        # self.boxBlurTransformAct = QAction(QtGui.QIcon('img/boxBlur.png'), '&Box Blur', self)
        # self.boxBlurTransformAct.setStatusTip('Box Blur')
        # self.boxBlurTransformAct.triggered.connect(self.applyBoxBlur)
        # # self.boxBlurTransformAct.setEnabled(False)
        # self.blurMenu.addAction(self.boxBlurTransformAct)
        #
        # self.verticalMotionBlurTransformAct = QAction(QtGui.QIcon('img/verticalMotionBlur.png'),
        #                                               '&Vertical Motion Blur', self)
        # self.verticalMotionBlurTransformAct.setStatusTip('Vertical Motion Blur')
        # self.verticalMotionBlurTransformAct.triggered.connect(self.createVerticalMotionBlurWindow)
        # # self.verticalMotionBlurTransformAct.setEnabled(False)
        # self.blurMenu.addAction(self.verticalMotionBlurTransformAct)
        #
        # self.horizontalMotionBlurTransformAct = QAction(QtGui.QIcon('img/horizontalMotionBlur.png'),
        #                                                 '&Horizontal Motion Blur', self)
        # self.horizontalMotionBlurTransformAct.setStatusTip('Horizontal Motion Blur')
        # self.horizontalMotionBlurTransformAct.triggered.connect(self.createHorizontalMotionBlurWindow)
        # # self.horizontalMotionBlurTransformAct.setEnabled(False)
        # self.blurMenu.addAction(self.horizontalMotionBlurTransformAct)
        #
        # # border
        # self.borderAct = QAction(QtGui.QIcon('img/border.png'), '&Border', self)
        # self.borderAct.setStatusTip('Border')
        # self.borderAct.setShortcut('Ctrl+B')
        # self.borderAct.triggered.connect(self.createBorderDialog)
        # self.borderAct.setEnabled(False)
        # transformMenu.addAction(self.borderAct)
        self.originalAct = QAction("Original", self)
        self.originalAct.setStatusTip('Original Image')
        self.originalAct.setEnabled(False)
        self.originalAct.triggered.connect(self.performOriginalImage)

        helpAct = QAction("Help", self)
        # helpAct.triggered.connect()

        self.setMenuBar(menuBar)
        menuBar.addMenu(fileMenu)
        menuBar.addMenu(editMenu)
        menuBar.addMenu(viewMenu)
        menuBar.addAction(self.originalAct)
        menuBar.addAction(helpAct)
        # menuBar.addMenu(transformMenu)

    def createLeftToolBar(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")

        # self.leftwidget = QWidget()  # 基础窗口控件
        # # self.leftwidget.setGeometry(0, 0, 38, 38 * 11)
        # self.baselayout.setContentsMargins(0, 0, 0, 0)
        # self.baselayout.addWidget(self.leftwidget)
        self.vboxLayout = QVBoxLayout()  # 设置垂直布局
        self.leftwidget.setLayout(self.vboxLayout)

        # self.listWidget = QListWidget(self)
        # self.baselayout.addWidget(self.listWidget)

        # 设置Button
        # grayScaleAct = QAction('Gray Scale', self)
        # grayScaleAct.setStatusTip('Gray Scale')
        # grayScaleAct.triggered.connect(self.performGrayScale)
        #
        # flipAct = QAction(QtGui.QIcon('img/flip.png'), 'Flip', self)
        # flipAct.setStatusTip('Flip')
        # flipAct.triggered.connect(self.showFlipPanel)
        #
        # separator1 = QAction(QtGui.QIcon('img/rotation.png'), 'Separator', self)
        # separator1.setSeparator(True)
        #
        # rotationAct = QAction(QtGui.QIcon('img/rotation.png'), 'Rotation', self)
        # rotationAct.setStatusTip('Rotation')
        # rotationAct.triggered.connect(self.showRotatePanel)
        #
        # cropAct = QAction(QtGui.QIcon('img/crop.png'), 'Crop', self)
        # cropAct.setStatusTip('Crop')
        # cropAct.triggered.connect(self.showCropPanel)
        #
        # overlayAct = QAction(QtGui.QIcon('img/overlay.png'), 'Overlay', self)
        # overlayAct.setStatusTip('Overlay')
        # overlayAct.triggered.connect(self.showOverlayPanel)
        #
        # separator2 = QAction(QtGui.QIcon('img/rotation.png'), 'Separator', self)
        # separator2.setSeparator(True)
        #
        # invertAct = QAction(QtGui.QIcon('img/invert.png'), 'Invert', self)
        # invertAct.setStatusTip('Invert')
        # invertAct.triggered.connect(self.applyInvert)
        #
        # textAct = QAction(QtGui.QIcon('img/text.png'), 'Text', self)
        # textAct.setStatusTip('Text')
        # textAct.triggered.connect(self.showTextPanel)
        #
        # separator3 = QAction(QtGui.QIcon('img/rotation.png'), 'Separator', self)
        # separator3.setSeparator(True)
        #
        # sharpAct = QAction(QtGui.QIcon('img/sharp.png'), 'Sharp', self)
        # sharpAct.setStatusTip('Sharp')
        # sharpAct.triggered.connect(self.applySharpening)
        #
        # unsharpAct = QAction(QtGui.QIcon('img/unsharp.png'), 'Unsharp', self)
        # unsharpAct.setStatusTip('Unsharp')
        # unsharpAct.triggered.connect(self.applyUnsharpening)
        #
        # laplaceAct = QAction(QtGui.QIcon('img/laplace.png'), 'Laplace', self)
        # laplaceAct.setStatusTip('Laplace')
        # laplaceAct.triggered.connect(self.applyLaplace)
        #
        # separator4 = QAction(QtGui.QIcon('img/rotation.png'), 'Separator', self)
        # separator4.setSeparator(True)
        #
        # watermarkAct = QAction(QtGui.QIcon('img/watermark.png'), 'Watermark', self)
        # watermarkAct.setStatusTip('Watermark')
        # watermarkAct.triggered.connect(self.showWatermarkPanel)

        self.toolButton = QToolButton(self.leftwidget)
        self.toolButton.setIcon(QtGui.QIcon('img/gs.png'))
        self.toolButton.setStatusTip('Gray Scale')
        self.toolButton.setFixedSize(38, 38)
        # self.toolButton.addAction(grayScaleAct)
        self.toolButton.clicked.connect(self.performGrayScale)
        self.toolButton1 = QToolButton(self.leftwidget)
        self.toolButton1.setIcon(QtGui.QIcon('img/flip.png'))
        self.toolButton1.setStatusTip('Flip')
        self.toolButton1.setFixedSize(38, 38)
        self.toolButton1.clicked.connect(self.showFlipPanel)

        self.toolButton3 = QToolButton(self.leftwidget)
        self.toolButton3.setIcon(QtGui.QIcon('img/rotation.png'))
        self.toolButton3.setStatusTip('Rotation')
        self.toolButton3.setFixedSize(38, 38)
        self.toolButton3.clicked.connect(self.showRotatePanel)
        self.toolButton4 = QToolButton(self.leftwidget)
        self.toolButton4.setIcon(QtGui.QIcon('img/crop.png'))
        self.toolButton4.setStatusTip('Crop')
        self.toolButton4.setFixedSize(38, 38)
        self.toolButton4.clicked.connect(self.showCropPanel)
        self.toolButton5 = QToolButton(self.leftwidget)
        self.toolButton5.setIcon(QtGui.QIcon('img/overlay.png'))
        self.toolButton5.setStatusTip('Overlay')
        self.toolButton5.setFixedSize(38, 38)
        self.toolButton5.clicked.connect(self.showOverlayPanel)

        self.toolButton7 = QToolButton(self.leftwidget)
        self.toolButton7.setIcon(QtGui.QIcon('img/invert.png'))
        self.toolButton7.setStatusTip('Invert')
        self.toolButton7.setFixedSize(38, 38)
        self.toolButton7.clicked.connect(self.applyInvert)
        self.toolButton8 = QToolButton(self.leftwidget)
        self.toolButton8.setIcon(QtGui.QIcon('img/text.png'))
        self.toolButton8.setStatusTip('Text')
        self.toolButton8.setFixedSize(38, 38)
        self.toolButton8.clicked.connect(self.showTextPanel)

        self.toolButton10 = QToolButton(self.leftwidget)
        self.toolButton10.setIcon(QtGui.QIcon('img/sharp.png'))
        self.toolButton10.setStatusTip('Sharp')
        self.toolButton10.setFixedSize(38, 38)
        self.toolButton10.clicked.connect(self.applySharpening)
        self.toolButton11 = QToolButton(self.leftwidget)
        self.toolButton11.setIcon(QtGui.QIcon('img/unsharp.png'))
        self.toolButton11.setStatusTip('Unsharp')
        self.toolButton11.setFixedSize(38, 38)
        self.toolButton11.clicked.connect(self.applyUnsharpening)
        self.toolButton12 = QToolButton(self.leftwidget)
        self.toolButton12.setIcon(QtGui.QIcon('img/laplace.png'))
        self.toolButton12.setStatusTip('Laplace')
        self.toolButton12.setFixedSize(38, 38)
        self.toolButton12.clicked.connect(self.applyLaplace)

        self.toolButton14 = QToolButton(self.leftwidget)
        self.toolButton14.setIcon(QtGui.QIcon('img/watermark.png'))
        self.toolButton14.setStatusTip('Watermark')
        self.toolButton14.setFixedSize(38, 38)
        self.toolButton14.clicked.connect(self.showWatermarkPanel)

        self.vboxLayout.addWidget(self.toolButton)
        self.vboxLayout.addWidget(self.toolButton1)

        self.vboxLayout.addWidget(self.toolButton3)
        self.vboxLayout.addWidget(self.toolButton4)
        self.vboxLayout.addWidget(self.toolButton5)

        self.vboxLayout.addWidget(self.toolButton7)
        self.vboxLayout.addWidget(self.toolButton8)

        self.vboxLayout.addWidget(self.toolButton10)
        self.vboxLayout.addWidget(self.toolButton11)
        self.vboxLayout.addWidget(self.toolButton12)

        self.vboxLayout.addWidget(self.toolButton14)
        self.vboxLayout.addStretch(0)
        self.vboxLayout.setSpacing(0)
        self.vboxLayout.setContentsMargins(0, 0, 0, 0)

    def performZoomIn(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(22)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performZoomOut(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(23)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performUndo(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        undoStack = self.images["undoStack"]
        redoStack = self.images["redoStack"]
        redoStack.append(undoStack.pop())
        imagePath = undoStack[len(undoStack) - 1]
        self.images["path"] = imagePath
        self.images["filterImageName"] = ""
        self.update()

    def performRedo(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        undoStack = self.images["undoStack"]
        redoStack = self.images["redoStack"]
        undoStack.append(redoStack.pop())
        imagePath = undoStack[len(undoStack) - 1]
        self.images["path"] = imagePath
        self.images["filterImageName"] = ""
        self.update()

    def resetResizeValues(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.resizeHeightTextArea.setText("0")
        self.resizeWidthTextArea.setText("0")

    def setBrightnessFactor(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.brightnessFactor = self.brightnessSlider.value()

    def setContrastFactor(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.contrastFactor = self.contrastSlider.value()

    def resizeImage(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(12)
        self.worker.signal.connect(self.taskFinished)
        self.resizeDialog.close()
        self.worker.start()
        self.showLoader()

    def setBrightness(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(2)
        self.worker.signal.connect(self.taskFinished)
        self.brightnessDialog.close()
        self.worker.start()
        self.showLoader()

    def setContrast(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(3)
        self.worker.signal.connect(self.taskFinished)
        self.contrastDialog.close()
        self.worker.start()
        self.showLoader()

    def createBorderDialog(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # open window for border setup
        self.normalBorderPanel1Activator = False
        self.normalBorderPanel2Activator = False

        self.replicateBorderPanel1Activator = False
        self.replicateBorderPanel2Activator = False

        self.replicate_standardRadioButton = None
        self.replicate_customRadioButton = None

        self.normal_standardRadioButton = None
        self.normal_customRadioButton = None

        self.borderDialog = QDialog(self)
        self.borderDialog.resize(300, 225)
        self.borderDialog.setWindowTitle("Set Border")

        borderTypeLabel = QLabel(self.borderDialog)
        borderTypeLabel.setText("Border-Type : ")
        borderTypeLabel.setGeometry(50, 10, 100, 50)
        borderTypeLabel.show()

        self.borderTypeList = QComboBox(self.borderDialog)
        self.borderTypeList.addItem("Select", -1)
        self.borderTypeList.addItem("Normal", 0)
        self.borderTypeList.addItem("Replicate", 1)
        self.borderTypeList.activated.connect(self.showBorderPanels)
        self.borderTypeList.setEditable(False)
        self.borderTypeList.setGeometry(130, 25, 100, 20)
        self.borderTypeList.show()
        self.borderDialog.exec_()

    def showNormalStandardBorderPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.normalBorderPanel1Activator = True
        self.hideNormalCustomBorderPanel()
        self.normal_borderThicknessLabel.show()
        self.normal_borderThicknessTextArea.show()
        self.normal_borderThicknessPixelLabel.show()
        self.normal_borderColorLabel.show()
        self.normal_borderColorButton.show()
        self.normal_standardOKButton.show()
        self.normal_standardCancelButton.show()

    def showNormalCustomBorderPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.normalBorderPanel2Activator = True
        self.hideNormalStandardBorderPanel()
        self.normal_borderTopLabel.show()
        self.normal_borderTopTextArea.show()
        self.normal_borderBottomLabel.show()
        self.normal_borderBottomTextArea.show()
        self.normal_borderLeftLabel.show()
        self.normal_borderLeftTextArea.show()
        self.normal_borderRightLabel.show()
        self.normal_borderRightTextArea.show()
        self.normal_customBorderColorLabel.show()
        self.normal_customBorderColorButton.show()
        self.normal_customOKButton.show()
        self.normal_customCancelButton.show()

    def hideNormalCustomBorderPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.normalBorderPanel2Activator = False
        self.normal_borderTopLabel.hide()
        self.normal_borderTopTextArea.hide()
        self.normal_borderBottomLabel.hide()
        self.normal_borderBottomTextArea.hide()
        self.normal_borderLeftLabel.hide()
        self.normal_borderLeftTextArea.hide()
        self.normal_borderRightLabel.hide()
        self.normal_borderRightTextArea.hide()
        self.normal_customBorderColorLabel.hide()
        self.normal_customBorderColorButton.hide()
        self.normal_customOKButton.hide()
        self.normal_customCancelButton.hide()

    def hideNormalStandardBorderPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.normalBorderPanel1Activator = False
        self.normal_borderThicknessLabel.hide()
        self.normal_borderThicknessTextArea.hide()
        self.normal_borderThicknessPixelLabel.hide()
        self.normal_borderColorLabel.hide()
        self.normal_borderColorButton.hide()
        self.normal_standardOKButton.hide()
        self.normal_standardCancelButton.hide()

    def showReplicateStandardBorderPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.replicateBorderPanel1Activator = True
        self.hideReplicateCustomBorderPanel()
        self.replicate_borderThicknessLabel.show()
        self.replicate_borderThicknessTextArea.show()
        self.replicate_borderThicknessPixelLabel.show()
        self.replicate_standardOKButton.show()
        self.replicate_standardCancelButton.show()

    def hideReplicateStandardBorderPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.replicateBorderPanel1Activator = False
        self.replicate_borderThicknessLabel.hide()
        self.replicate_borderThicknessTextArea.hide()
        self.replicate_borderThicknessPixelLabel.hide()
        self.replicate_standardOKButton.hide()
        self.replicate_standardCancelButton.hide()

    def showReplicateCustomBorderPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.replicateBorderPanel2Activator = True
        self.hideReplicateStandardBorderPanel()
        self.replicate_borderTopLabel.show()
        self.replicate_borderTopTextArea.show()
        self.replicate_borderBottomLabel.show()
        self.replicate_borderBottomTextArea.show()
        self.replicate_borderLeftLabel.show()
        self.replicate_borderLeftTextArea.show()
        self.replicate_borderRightLabel.show()
        self.replicate_borderRightTextArea.show()
        self.replicate_customOKButton.show()
        self.replicate_customCancelButton.show()

    def hideReplicateCustomBorderPanel(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.replicateBorderPanel2Activator = False
        self.replicate_borderTopLabel.hide()
        self.replicate_borderTopTextArea.hide()
        self.replicate_borderBottomLabel.hide()
        self.replicate_borderBottomTextArea.hide()
        self.replicate_borderLeftLabel.hide()
        self.replicate_borderLeftTextArea.hide()
        self.replicate_borderRightLabel.hide()
        self.replicate_borderRightTextArea.hide()
        self.replicate_customOKButton.hide()
        self.replicate_customCancelButton.hide()

    def resetBorderPanels(self, index):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if index == 1:
            if self.replicateBorderPanel1Activator:
                self.hideReplicateStandardBorderPanel()
            if self.replicateBorderPanel2Activator:
                self.hideReplicateCustomBorderPanel()

            if self.replicate_standardRadioButton != None:
                self.replicate_standardRadioButton.hide()
            if self.replicate_customRadioButton != None:
                self.replicate_customRadioButton.hide()
                self.separatorLine2.hide()

        elif index == 2:
            if self.normalBorderPanel1Activator:
                self.hideNormalStandardBorderPanel()
            if self.normalBorderPanel2Activator:
                self.hideNormalCustomBorderPanel()
            if self.normal_standardRadioButton != None:
                self.normal_standardRadioButton.hide()
            if self.normal_customRadioButton != None:
                self.normal_customRadioButton.hide()
                self.separatorLine1.hide()

    def open_normal_standard_color_chooser(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        color = QColorDialog.getColor()
        if color.isValid():
            borderColor = color.name()
            self.normal_borderColorButton.setText(borderColor)

    def validate_normal_standard_border_factors(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        color = self.normal_borderColorButton.text()
        errorMessage = ""
        hasError = False
        try:
            thickness = int(self.normal_borderThicknessTextArea.text())
        except:
            errorMessage += "Invalid thickness input. \r\n"
            hasError = True
        errorDialog = QtWidgets.QErrorMessage(self.borderDialog)
        if color.startswith("Choose"):
            hasError = True
            errorMessage += "Invalid color selection. \r\n"
        if hasError:
            errorDialog.showMessage(errorMessage)
            errorDialog.setWindowTitle(f"Invalid Input Error")
        else:
            operationString = dict()
            operationString["thickness"] = thickness
            operationString["isTuple"] = False
            operationString["color"] = color
            operationString["border-type"] = "normal_standard"

            self.worker.setTmApplication(self)
            self.worker.setOperation(26)
            self.worker.setOperationString(operationString)
            self.worker.signal.connect(self.taskFinished)
            self.worker.start()
            self.showLoader()

    def open_normal_custom_color_chooser(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        color = QColorDialog.getColor()
        if color.isValid():
            borderColor = color.name()
            self.normal_customBorderColorButton.setText(borderColor)
            # done done

    def validate_normal_custom_border_factors(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        color = self.normal_customBorderColorButton.text()
        hasError = False
        errorMessage = ""
        try:
            top = int(self.normal_borderTopTextArea.text())
        except:
            errorMessage += "Invalid top value input. \r\n"
            hasError = True
        try:
            bottom = int(self.normal_borderBottomTextArea.text())
        except:
            errorMessage += "Invalid bottom value input. \r\n"
            hasError = True
        try:
            left = int(self.normal_borderLeftTextArea.text())
        except:
            errorMessage += "Invalid left value input. \r\n"
            hasError = True
        try:
            right = int(self.normal_borderRightTextArea.text())
        except:
            errorMessage += "Invalid right value input. \r\n"
            hasError = True
        if len(color) == 0 or color.startswith("Choose..."):
            errorMessage = "Invalid color selection. \r\n"
            hasError = True
        if hasError:
            errorDialog = QtWidgets.QErrorMessage(self.borderDialog)
            errorDialog.showMessage(errorMessage)
            errorDialog.showWindowTitle("Invalid Input Error")
        else:
            operationString = dict()
            thickness = list()
            thickness.append(top)
            thickness.append(bottom)
            thickness.append(left)
            thickness.append(right)
            operationString["thickness"] = thickness
            operationString["isTuple"] = True
            operationString["color"] = color
            operationString["border-type"] = "normal_custom"

            self.worker.setTmApplication(self)
            self.worker.setOperation(26)
            self.worker.setOperationString(operationString)
            self.worker.signal.connect(self.taskFinished)
            self.worker.start()
            self.showLoader()

    def validate_replicate_standard_border_parameters(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        hasError = False
        errorMessage = ""
        try:
            thickness = int(self.replicate_borderThicknessTextArea.text())
        except:
            errorMessage += "Invalid thickness input. \r\n"
            hasError = True
        if hasError:
            errorDialog = QtWidgets.QErrorMessage(self.borderDialog)
            errorDialog.showMessage(errorMessage)
            errorDialog.showWindowTitle("Invalid Input Error")
        else:
            operationString = dict()
            operationString["thickness"] = thickness
            operationString["isTuple"] = True
            operationString["border-type"] = "replicate_standard"

            self.worker.setTmApplication(self)
            self.worker.setOperation(27)
            self.worker.setOperationString(operationString)
            self.worker.signal.connect(self.taskFinished)
            self.worker.start()
            self.showLoader()

    def validate_replicate_custom_border_parameters(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        hasError = False
        errorMessage = ""
        try:
            top = int(self.replicate_borderTopTextArea.text())
        except:
            errorMessage += "Invalid top value input. \r\n"
            hasError = True
        try:
            bottom = int(self.replicate_borderBottomTextArea.text())
        except:
            errorMessage += "Invalid bottom value input. \r\n"
            hasError = True
        try:
            left = int(self.replicate_borderLeftTextArea.text())
        except:
            errorMessage += "Invalid left value input. \r\n"
            hasError = True
        try:
            right = int(self.replicate_borderRightTextArea.text())
        except:
            errorMessage += "Invalid right value input. \r\n"
            hasError = True

        if hasError:
            errorDialog = QtWidgets.QErrorMessage(self.borderDialog)
            errorDialog.showMessage(errorMessage)
            errorDialog.showWindowTitle("Invalid Input Error")
        else:
            operationString = dict()
            thickness = list()
            thickness.append(top)
            thickness.append(bottom)
            thickness.append(left)
            thickness.append(right)
            operationString["thickness"] = thickness
            operationString["isTuple"] = True
            operationString["border-type"] = "replicate_custom"

            self.worker.setTmApplication(self)
            self.worker.setOperation(27)
            self.worker.setOperationString(operationString)
            self.worker.signal.connect(self.taskFinished)
            self.worker.start()
            self.showLoader()

    def showBorderPanels(self, index):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if index == 1:
            self.resetBorderPanels(index)
            # normal standard border
            self.normalBorderPanel1Activator = True
            self.normal_standardRadioButton = QRadioButton(self.borderDialog)
            self.normal_standardRadioButton.setText("Standard")
            self.normal_standardRadioButton.setChecked(True)
            self.normal_standardRadioButton.setGeometry(50, 50, 100, 50)
            self.normal_standardRadioButton.clicked.connect(self.showNormalStandardBorderPanel)
            self.normal_standardRadioButton.show()

            self.normal_customRadioButton = QRadioButton(self.borderDialog)
            self.normal_customRadioButton.setText("Custom")
            self.normal_customRadioButton.setGeometry(190, 50, 100, 50)
            self.normal_customRadioButton.clicked.connect(self.showNormalCustomBorderPanel)
            self.normal_customRadioButton.show()

            # separator line
            self.separatorLine1 = QFrame(self.borderDialog)
            self.separatorLine1.setFrameShape(QFrame.HLine)
            self.separatorLine1.setFrameShadow(QFrame.Sunken)
            self.separatorLine1.setGeometry(0, 70, 300, 50)
            self.separatorLine1.show()

            self.normal_borderThicknessLabel = QLabel(self.borderDialog)
            self.normal_borderThicknessLabel.setText("Thickness : ")
            self.normal_borderThicknessLabel.setGeometry(50, 100, 150, 50)
            self.normal_borderThicknessLabel.show()

            self.normal_borderThicknessTextArea = QLineEdit("", self.borderDialog)
            self.normal_borderThicknessTextArea.setGeometry(115, 115, 70, 20)
            self.normal_borderThicknessTextArea.show()

            self.normal_borderThicknessPixelLabel = QLabel(self.borderDialog)
            self.normal_borderThicknessPixelLabel.setText("Pixels")
            self.normal_borderThicknessPixelLabel.setGeometry(200, 117, 40, 15)
            self.normal_borderThicknessPixelLabel.setStyleSheet("background:white;")
            self.normal_borderThicknessPixelLabel.setAlignment(QtCore.Qt.AlignCenter)
            self.normal_borderThicknessPixelLabel.show()

            self.normal_borderColorLabel = QLabel(self.borderDialog)
            self.normal_borderColorLabel.setText("Color : ")
            self.normal_borderColorLabel.setGeometry(50, 135, 150, 50)
            self.normal_borderColorLabel.show()

            self.normal_borderColorButton = QPushButton("Choose...", self.borderDialog)
            self.normal_borderColorButton.setIcon(QIcon("img/choose.png"))
            self.normal_borderColorButton.clicked.connect(self.open_normal_standard_color_chooser)
            self.normal_borderColorButton.setGeometry(113, 150, 100, 23)
            self.normal_borderColorButton.show()

            self.normal_standardOKButton = QPushButton("OK", self.borderDialog)
            self.normal_standardOKButton.move(70, 190)
            self.normal_standardOKButton.setIcon(QIcon("img/ok.png"))
            self.normal_standardOKButton.clicked.connect(self.validate_normal_standard_border_factors)
            self.normal_standardOKButton.show()

            self.normal_standardCancelButton = QPushButton("Cancel", self.borderDialog)
            self.normal_standardCancelButton.move(170, 190)
            self.normal_standardCancelButton.setIcon(QIcon("img/cancel.png"))
            self.normal_standardCancelButton.clicked.connect(self.borderDialog.close)
            self.normal_standardCancelButton.show()

            # normal custom border

            self.normal_borderTopLabel = QLabel(self.borderDialog)
            self.normal_borderTopLabel.setText("Top : ")
            self.normal_borderTopLabel.setGeometry(30, 90, 150, 50)
            self.normal_borderTopLabel.hide()

            self.normal_borderTopTextArea = QLineEdit("", self.borderDialog)
            self.normal_borderTopTextArea.setGeometry(70, 107, 50, 18)
            self.normal_borderTopTextArea.hide()

            self.normal_borderBottomLabel = QLabel(self.borderDialog)
            self.normal_borderBottomLabel.setText("Bottom : ")
            self.normal_borderBottomLabel.setGeometry(160, 90, 150, 50)
            self.normal_borderBottomLabel.hide()

            self.normal_borderBottomTextArea = QLineEdit("", self.borderDialog)
            self.normal_borderBottomTextArea.setGeometry(210, 107, 50, 18)
            self.normal_borderBottomTextArea.hide()

            self.normal_borderLeftLabel = QLabel(self.borderDialog)
            self.normal_borderLeftLabel.setText("Left : ")
            self.normal_borderLeftLabel.setGeometry(30, 115, 150, 50)
            self.normal_borderLeftLabel.hide()

            self.normal_borderLeftTextArea = QLineEdit("", self.borderDialog)
            self.normal_borderLeftTextArea.setGeometry(70, 132, 50, 18)
            self.normal_borderLeftTextArea.hide()

            self.normal_borderRightLabel = QLabel(self.borderDialog)
            self.normal_borderRightLabel.setText("Right : ")
            self.normal_borderRightLabel.setGeometry(160, 115, 150, 50)
            self.normal_borderRightLabel.hide()

            self.normal_borderRightTextArea = QLineEdit("", self.borderDialog)
            self.normal_borderRightTextArea.setGeometry(210, 132, 50, 18)
            self.normal_borderRightTextArea.hide()

            self.normal_customBorderColorLabel = QLabel(self.borderDialog)
            self.normal_customBorderColorLabel.setText("Color : ")
            self.normal_customBorderColorLabel.setGeometry(30, 143, 150, 50)
            self.normal_customBorderColorLabel.hide()

            self.normal_customBorderColorButton = QPushButton("Choose...", self.borderDialog)
            self.normal_customBorderColorButton.setIcon(QIcon("img/choose.png"))
            self.normal_customBorderColorButton.clicked.connect(self.open_normal_custom_color_chooser)
            self.normal_customBorderColorButton.setGeometry(70, 158, 100, 20)
            self.normal_customBorderColorButton.hide()

            self.normal_customOKButton = QPushButton("OK", self.borderDialog)
            self.normal_customOKButton.move(70, 190)
            self.normal_customOKButton.setIcon(QIcon("img/ok.png"))
            self.normal_customOKButton.clicked.connect(self.validate_normal_custom_border_factors)
            self.normal_customOKButton.hide()

            self.normal_customCancelButton = QPushButton("Cancel", self.borderDialog)
            self.normal_customCancelButton.move(170, 190)
            self.normal_customCancelButton.setIcon(QIcon("img/cancel.png"))
            self.normal_customCancelButton.clicked.connect(self.borderDialog.close)
            self.normal_customCancelButton.hide()

        elif index == 2:
            self.resetBorderPanels(index)
            # replicate standard border
            self.replicateBorderPanel1Activator = True
            self.replicate_standardRadioButton = QRadioButton(self.borderDialog)
            self.replicate_standardRadioButton.setText("Standard")
            self.replicate_standardRadioButton.setChecked(True)
            self.replicate_standardRadioButton.setGeometry(50, 50, 100, 50)
            self.replicate_standardRadioButton.clicked.connect(self.showReplicateStandardBorderPanel)
            self.replicate_standardRadioButton.show()

            self.replicate_customRadioButton = QRadioButton(self.borderDialog)
            self.replicate_customRadioButton.setText("Custom")
            self.replicate_customRadioButton.setGeometry(190, 50, 100, 50)
            self.replicate_customRadioButton.clicked.connect(self.showReplicateCustomBorderPanel)
            self.replicate_customRadioButton.show()

            # separator line
            self.separatorLine2 = QFrame(self.borderDialog)
            self.separatorLine2.setFrameShape(QFrame.HLine)
            self.separatorLine2.setFrameShadow(QFrame.Sunken)
            self.separatorLine2.setGeometry(0, 70, 300, 50)
            self.separatorLine2.show()

            self.replicate_borderThicknessLabel = QLabel(self.borderDialog)
            self.replicate_borderThicknessLabel.setText("Thickness : ")
            self.replicate_borderThicknessLabel.setGeometry(50, 100, 150, 50)
            self.replicate_borderThicknessLabel.show()

            self.replicate_borderThicknessTextArea = QLineEdit("", self.borderDialog)
            self.replicate_borderThicknessTextArea.setGeometry(115, 115, 70, 20)
            self.replicate_borderThicknessTextArea.show()

            self.replicate_borderThicknessPixelLabel = QLabel(self.borderDialog)
            self.replicate_borderThicknessPixelLabel.setText("Pixels")
            self.replicate_borderThicknessPixelLabel.setGeometry(200, 117, 40, 15)
            self.replicate_borderThicknessPixelLabel.setStyleSheet("background:white;")
            self.replicate_borderThicknessPixelLabel.setAlignment(QtCore.Qt.AlignCenter)
            self.replicate_borderThicknessPixelLabel.show()

            self.replicate_standardOKButton = QPushButton("OK", self.borderDialog)
            self.replicate_standardOKButton.move(70, 190)
            self.replicate_standardOKButton.setIcon(QIcon("img/ok.png"))
            self.replicate_standardOKButton.clicked.connect(self.validate_replicate_standard_border_parameters)
            self.replicate_standardOKButton.show()

            self.replicate_standardCancelButton = QPushButton("Cancel", self.borderDialog)
            self.replicate_standardCancelButton.move(170, 190)
            self.replicate_standardCancelButton.setIcon(QIcon("img/cancel.png"))
            self.replicate_standardCancelButton.clicked.connect(self.borderDialog.close)
            self.replicate_standardCancelButton.show()

            # replicate custom border

            self.replicate_borderTopLabel = QLabel(self.borderDialog)
            self.replicate_borderTopLabel.setText("Top : ")
            self.replicate_borderTopLabel.setGeometry(30, 90, 150, 50)
            self.replicate_borderTopLabel.hide()

            self.replicate_borderTopTextArea = QLineEdit("", self.borderDialog)
            self.replicate_borderTopTextArea.setGeometry(70, 107, 50, 18)
            self.replicate_borderTopTextArea.hide()

            self.replicate_borderBottomLabel = QLabel(self.borderDialog)
            self.replicate_borderBottomLabel.setText("Bottom : ")
            self.replicate_borderBottomLabel.setGeometry(160, 90, 150, 50)
            self.replicate_borderBottomLabel.hide()

            self.replicate_borderBottomTextArea = QLineEdit("", self.borderDialog)
            self.replicate_borderBottomTextArea.setGeometry(210, 107, 50, 18)
            self.replicate_borderBottomTextArea.hide()

            self.replicate_borderLeftLabel = QLabel(self.borderDialog)
            self.replicate_borderLeftLabel.setText("Left : ")
            self.replicate_borderLeftLabel.setGeometry(30, 130, 150, 50)
            self.replicate_borderLeftLabel.hide()

            self.replicate_borderLeftTextArea = QLineEdit("", self.borderDialog)
            self.replicate_borderLeftTextArea.setGeometry(70, 147, 50, 18)
            self.replicate_borderLeftTextArea.hide()

            self.replicate_borderRightLabel = QLabel(self.borderDialog)
            self.replicate_borderRightLabel.setText("Right : ")
            self.replicate_borderRightLabel.setGeometry(160, 130, 150, 50)
            self.replicate_borderRightLabel.hide()

            self.replicate_borderRightTextArea = QLineEdit("", self.borderDialog)
            self.replicate_borderRightTextArea.setGeometry(210, 147, 50, 18)
            self.replicate_borderRightTextArea.hide()

            self.replicate_customOKButton = QPushButton("OK", self.borderDialog)
            self.replicate_customOKButton.move(70, 190)
            self.replicate_customOKButton.setIcon(QIcon("img/ok.png"))
            self.replicate_customOKButton.clicked.connect(self.validate_replicate_custom_border_parameters)
            self.replicate_customOKButton.hide()

            self.replicate_customCancelButton = QPushButton("Cancel", self.borderDialog)
            self.replicate_customCancelButton.move(170, 190)
            self.replicate_customCancelButton.setIcon(QIcon("img/cancel.png"))
            self.replicate_customCancelButton.clicked.connect(self.borderDialog.close)
            self.replicate_customCancelButton.hide()

    def createBrightnessWindow(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # open window for brightness setup
        self.brightnessDialog = QDialog(self)
        self.brightnessDialog.resize(300, 150)
        self.brightnessDialog.setWindowTitle("Set Brightness")

        label = QLabel(self.brightnessDialog)
        label.setText("Brightness: ")
        label.setGeometry(60, 16, 100, 30)

        self.brightnessSlider = QSlider(Qt.Horizontal, self.brightnessDialog)
        self.brightnessSlider.setFocusPolicy(Qt.NoFocus)
        self.brightnessSlider.setTickPosition(QSlider.TicksAbove)
        self.brightnessSlider.setTickInterval(10)
        self.brightnessSlider.setSingleStep(10)
        self.brightnessSlider.setValue(50)
        self.brightnessSlider.setGeometry(60, 51, 170, 50)
        self.brightnessSlider.valueChanged.connect(self.setBrightnessFactor)

        self.brightnessbutton = QPushButton("OK", self.brightnessDialog)
        self.brightnessbutton.move(110, 111)
        self.brightnessbutton.clicked.connect(self.setBrightness)

        self.brightnessDialog.exec_()

    def createResizeWindow(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")

        imageSize = str(self.images["size"])
        imageHeight = str(self.images["height"])
        imageWidth = str(self.images["width"])

        self.resizeDialog = QDialog(self)
        self.resizeDialog.resize(400, 200)
        self.resizeDialog.setWindowTitle("Resize Image")

        sizeTitleLabel = QLabel(self.resizeDialog)
        sizeTitleLabel.setText("Image Size: ")
        sizeTitleLabel.setGeometry(40, 16, 100, 30)

        sizeLabel = QLabel(self.resizeDialog)
        sizeLabel.setText(imageSize + " MB")
        sizeLabel.setGeometry(110, 16, 100, 30)

        dimensionsTitleLabel = QLabel(self.resizeDialog)
        dimensionsTitleLabel.setText("Dimensions: ")
        dimensionsTitleLabel.setGeometry(40, 46, 100, 30)

        dimensionsLabel = QLabel(self.resizeDialog)
        dimensionsLabel.setText(imageHeight + "x" + imageWidth)
        dimensionsLabel.setGeometry(110, 46, 100, 30)

        heightTitleLabel = QLabel(self.resizeDialog)
        heightTitleLabel.setText("Height: ")
        heightTitleLabel.setGeometry(40, 90, 100, 30)

        self.resizeHeightTextArea = QLineEdit(self.resizeDialog)
        self.resizeHeightTextArea.setText(imageHeight)
        self.resizeHeightTextArea.setGeometry(90, 95, 80, 20)
        self.resizeHeightTextArea.textChanged.connect(self.updateResizeParameters)

        pixelLabel1 = QLabel(self.resizeDialog)
        pixelLabel1.setText("Pixels")
        pixelLabel1.setGeometry(190, 95, 80, 20)
        pixelLabel1.setStyleSheet("background-color:white;")
        pixelLabel1.setAlignment(QtCore.Qt.AlignCenter)

        widthTitleLabel = QLabel(self.resizeDialog)
        widthTitleLabel.setText("Width: ")
        widthTitleLabel.setGeometry(40, 120, 100, 30)

        self.resizeWidthTextArea = QLineEdit(self.resizeDialog)
        self.resizeWidthTextArea.setText(imageWidth)
        self.resizeWidthTextArea.setGeometry(90, 125, 80, 20)
        self.resizeWidthTextArea.textChanged.connect(self.updateResizeParameters)

        pixelLabel2 = QLabel(self.resizeDialog)
        pixelLabel2.setText("Pixels")
        pixelLabel2.setGeometry(190, 125, 80, 20)
        pixelLabel2.setStyleSheet("background-color:white;")
        pixelLabel2.setAlignment(QtCore.Qt.AlignCenter)

        self.resizeOKButton = QPushButton("OK", self.resizeDialog)
        self.resizeOKButton.move(310, 21)
        self.resizeOKButton.clicked.connect(self.resizeImage)

        self.resizeCancelButton = QPushButton("Cancel", self.resizeDialog)
        self.resizeCancelButton.move(310, 55)
        self.resizeCancelButton.clicked.connect(self.resizeDialog.close)

        self.resizeResetButton = QPushButton("Reset", self.resizeDialog)
        self.resizeResetButton.move(310, 81)
        self.resizeResetButton.clicked.connect(self.resetResizeValues)

        self.resizeDialog.exec_()

    def applyBoxBlur(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(16)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def applyVerticalMotionBlur(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        radius = 10
        if self.verticalMotionBlur10RadioButton.isChecked():
            radius = 10
        elif self.verticalMotionBlur20RadioButton.isChecked():
            radius = 20
        elif self.verticalMotionBlur50RadioButton.isChecked():
            radius = 50
        elif self.verticalMotionBlur100RadioButton.isChecked():
            radius = 100

        self.worker.setTmApplication(self)
        self.worker.setOperation(14)
        self.worker.setRadius(radius)
        self.verticalMotionBlurDialog.close()
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def applyLaplace(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(20)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def applyUnsharpening(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(19)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def applySharpening(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(18)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def applyInvert(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker.setTmApplication(self)
        self.worker.setOperation(17)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def applyGuassianBlur(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        radius = 2
        if self.gaussian20RadioButton.isChecked():
            radius = 2
        elif self.gaussian40RadioButton.isChecked():
            radius = 4
        elif self.gaussian60RadioButton.isChecked():
            radius = 6

        self.worker.setTmApplication(self)
        self.worker.setOperation(13)
        self.worker.setRadius(radius)
        self.guassianBlurDialog.close()
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def showGaussianPreview(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if self.gaussian20RadioButton.isChecked():
            self.gaussianBlurPreview.setPixmap(QPixmap("img/previews/gaussian20.jpg"))
        elif self.gaussian40RadioButton.isChecked():
            self.gaussianBlurPreview.setPixmap(QPixmap("img/previews/gaussian40.jpg"))
        elif self.gaussian60RadioButton.isChecked():
            self.gaussianBlurPreview.setPixmap(QPixmap("img/previews/gaussian60.jpg"))

    def showVerticalMotionBlurPreview(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if self.verticalMotionBlur10RadioButton.isChecked():
            self.verticalMotionBlurPreview.setPixmap(QPixmap("img/previews/carVerticalMotion10.jpg"))
        elif self.verticalMotionBlur20RadioButton.isChecked():
            self.verticalMotionBlurPreview.setPixmap(QPixmap("img/previews/carVerticalMotion20.jpg"))
        elif self.verticalMotionBlur50RadioButton.isChecked():
            self.verticalMotionBlurPreview.setPixmap(QPixmap("img/previews/carVerticalMotion50.jpg"))
        elif self.verticalMotionBlur100RadioButton.isChecked():
            self.verticalMotionBlurPreview.setPixmap(QPixmap("img/previews/carVerticalMotion100.jpg"))

    def showHorizontalMotionBlurPreview(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        if self.horizontalMotionBlur10RadioButton.isChecked():
            self.horizontalMotionBlurPreview.setPixmap(QPixmap("img/previews/mountainHorizontalMotion10.jpg"))
        elif self.horizontalMotionBlur20RadioButton.isChecked():
            self.horizontalMotionBlurPreview.setPixmap(QPixmap("img/previews/mountainHorizontalMotion20.jpg"))
        elif self.horizontalMotionBlur50RadioButton.isChecked():
            self.horizontalMotionBlurPreview.setPixmap(QPixmap("img/previews/mountainHorizontalMotion50.jpg"))
        elif self.horizontalMotionBlur100RadioButton.isChecked():
            self.horizontalMotionBlurPreview.setPixmap(QPixmap("img/previews/mountainHorizontalMotion100.jpg"))

    def applyHorizontalMotionBlur(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        radius = 10
        if self.horizontalMotionBlur10RadioButton.isChecked():
            radius = 10
        elif self.horizontalMotionBlur20RadioButton.isChecked():
            radius = 20
        elif self.horizontalMotionBlur50RadioButton.isChecked():
            radius = 50
        elif self.horizontalMotionBlur100RadioButton.isChecked():
            radius = 100

        self.worker.setTmApplication(self)
        self.worker.setOperation(15)
        self.worker.setRadius(radius)
        self.horizontalMotionBlurDialog.close()
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def createHorizontalMotionBlurWindow(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # open window for guassian blur setup
        self.horizontalMotionBlurDialog = QDialog(self)
        self.horizontalMotionBlurDialog.resize(450, 320)
        self.horizontalMotionBlurDialog.setWindowTitle("Set Motion Blur (%) - Horizontally")

        self.horizontalMotionBlurPreview = QLabel(self.horizontalMotionBlurDialog)
        self.horizontalMotionBlurPreview.setPixmap(QPixmap("img/previews/mountainHorizontalMotion10.jpg"))
        self.horizontalMotionBlurPreview.setGeometry(15, 5, 420, 200)
        self.horizontalMotionBlurPreview.setStyleSheet("border:1px solid black;")

        self.horizontalMotionBlur10RadioButton = QRadioButton(self.horizontalMotionBlurDialog)
        self.horizontalMotionBlur10RadioButton.setGeometry(55, 205, 80, 80)
        self.horizontalMotionBlur10RadioButton.setText("10% Blur")
        self.horizontalMotionBlur10RadioButton.setChecked(True)
        self.horizontalMotionBlur10RadioButton.clicked.connect(self.showHorizontalMotionBlurPreview)
        self.horizontalMotionBlur10RadioButton.show()

        self.horizontalMotionBlur20RadioButton = QRadioButton(self.horizontalMotionBlurDialog)
        self.horizontalMotionBlur20RadioButton.setGeometry(55 + 80 + 20, 205, 80, 80)
        self.horizontalMotionBlur20RadioButton.setText("20% Blur")
        self.horizontalMotionBlur20RadioButton.clicked.connect(self.showHorizontalMotionBlurPreview)
        self.horizontalMotionBlur20RadioButton.show()

        self.horizontalMotionBlur50RadioButton = QRadioButton(self.horizontalMotionBlurDialog)
        self.horizontalMotionBlur50RadioButton.setGeometry(55 + 80 + 55 + 55 + 15, 205, 80, 80)
        self.horizontalMotionBlur50RadioButton.setText("50% Blur")
        self.horizontalMotionBlur50RadioButton.clicked.connect(self.showHorizontalMotionBlurPreview)
        self.horizontalMotionBlur50RadioButton.show()

        self.horizontalMotionBlur100RadioButton = QRadioButton(self.horizontalMotionBlurDialog)
        self.horizontalMotionBlur100RadioButton.setGeometry(55 + 80 + 55 + 55 + 80 + 35, 205, 80, 80)
        self.horizontalMotionBlur100RadioButton.setText("100% Blur")
        self.horizontalMotionBlur100RadioButton.clicked.connect(self.showHorizontalMotionBlurPreview)
        self.horizontalMotionBlur100RadioButton.show()

        self.okButton = QPushButton("Apply", self.horizontalMotionBlurDialog)
        self.okButton.move(140, 280)
        self.okButton.clicked.connect(self.applyHorizontalMotionBlur)

        self.cancelButton = QPushButton("Cancel", self.horizontalMotionBlurDialog)
        self.cancelButton.move(220, 280)
        self.cancelButton.clicked.connect(self.horizontalMotionBlurDialog.close)

        self.horizontalMotionBlurDialog.exec_()

    def createVerticalMotionBlurWindow(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # open window for guassian blur setup
        self.verticalMotionBlurDialog = QDialog(self)
        self.verticalMotionBlurDialog.resize(450, 320)
        self.verticalMotionBlurDialog.setWindowTitle("Set Motion Blur (%) - Vertically")

        self.verticalMotionBlurPreview = QLabel(self.verticalMotionBlurDialog)
        self.verticalMotionBlurPreview.setPixmap(QPixmap("img/previews/carVerticalMotion10.jpg"))
        self.verticalMotionBlurPreview.setGeometry(15, 5, 420, 200)
        self.verticalMotionBlurPreview.setStyleSheet("border:1px solid black;")

        self.verticalMotionBlur10RadioButton = QRadioButton(self.verticalMotionBlurDialog)
        self.verticalMotionBlur10RadioButton.setGeometry(55, 205, 80, 80)
        self.verticalMotionBlur10RadioButton.setText("10% Blur")
        self.verticalMotionBlur10RadioButton.setChecked(True)
        self.verticalMotionBlur10RadioButton.clicked.connect(self.showVerticalMotionBlurPreview)
        self.verticalMotionBlur10RadioButton.show()

        self.verticalMotionBlur20RadioButton = QRadioButton(self.verticalMotionBlurDialog)
        self.verticalMotionBlur20RadioButton.setGeometry(55 + 80 + 20, 205, 80, 80)
        self.verticalMotionBlur20RadioButton.setText("20% Blur")
        self.verticalMotionBlur20RadioButton.clicked.connect(self.showVerticalMotionBlurPreview)
        self.verticalMotionBlur20RadioButton.show()

        self.verticalMotionBlur50RadioButton = QRadioButton(self.verticalMotionBlurDialog)
        self.verticalMotionBlur50RadioButton.setGeometry(55 + 80 + 55 + 55 + 15, 205, 80, 80)
        self.verticalMotionBlur50RadioButton.setText("50% Blur")
        self.verticalMotionBlur50RadioButton.clicked.connect(self.showVerticalMotionBlurPreview)
        self.verticalMotionBlur50RadioButton.show()

        self.verticalMotionBlur100RadioButton = QRadioButton(self.verticalMotionBlurDialog)
        self.verticalMotionBlur100RadioButton.setGeometry(55 + 80 + 55 + 55 + 80 + 35, 205, 80, 80)
        self.verticalMotionBlur100RadioButton.setText("100% Blur")
        self.verticalMotionBlur100RadioButton.clicked.connect(self.showVerticalMotionBlurPreview)
        self.verticalMotionBlur100RadioButton.show()

        self.okButton = QPushButton("Apply", self.verticalMotionBlurDialog)
        self.okButton.move(140, 280)
        self.okButton.clicked.connect(self.applyVerticalMotionBlur)

        self.okButton = QPushButton("Cancel", self.verticalMotionBlurDialog)
        self.okButton.move(220, 280)
        self.okButton.clicked.connect(self.verticalMotionBlurDialog.close)

        self.verticalMotionBlurDialog.exec_()

    def createGuassianBlurWindow(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # open window for guassian blur setup
        self.guassianBlurDialog = QDialog(self)
        self.guassianBlurDialog.resize(450, 320)
        self.guassianBlurDialog.setWindowTitle("Set Guassian Blur Percentage (%)")

        self.gaussianBlurPreview = QLabel(self.guassianBlurDialog)
        self.gaussianBlurPreview.setPixmap(QPixmap("img/previews/gaussian20.jpg"))
        self.gaussianBlurPreview.setGeometry(15, 5, 420, 200)
        self.gaussianBlurPreview.setStyleSheet("border:1px solid black;")

        self.gaussian20RadioButton = QRadioButton(self.guassianBlurDialog)
        self.gaussian20RadioButton.setGeometry(55, 205, 80, 80)
        self.gaussian20RadioButton.setText("20% Blur")
        self.gaussian20RadioButton.setChecked(True)
        self.gaussian20RadioButton.clicked.connect(self.showGaussianPreview)
        self.gaussian20RadioButton.show()

        self.gaussian40RadioButton = QRadioButton(self.guassianBlurDialog)
        self.gaussian40RadioButton.setGeometry(55 + 35 + 80, 205, 80, 80)
        self.gaussian40RadioButton.setText("40% Blur")
        self.gaussian40RadioButton.clicked.connect(self.showGaussianPreview)
        self.gaussian40RadioButton.show()

        self.gaussian60RadioButton = QRadioButton(self.guassianBlurDialog)
        self.gaussian60RadioButton.setGeometry(55 + 35 + 80 + 35 + 80, 205, 80, 80)
        self.gaussian60RadioButton.setText("60% Blur")
        self.gaussian60RadioButton.clicked.connect(self.showGaussianPreview)
        self.gaussian60RadioButton.show()

        self.okButton = QPushButton("Apply", self.guassianBlurDialog)
        self.okButton.move(140, 280)
        self.okButton.clicked.connect(self.applyGuassianBlur)

        self.okButton = QPushButton("Cancel", self.guassianBlurDialog)
        self.okButton.move(220, 280)
        self.okButton.clicked.connect(self.guassianBlurDialog.close)

        self.guassianBlurDialog.exec_()

    def createContrastWindow(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # open window for contrast setup
        self.contrastDialog = QDialog(self)
        self.contrastDialog.resize(300, 150)
        self.contrastDialog.setWindowTitle("Set Contrast")

        label = QLabel(self.contrastDialog)
        label.setText("Contrast: ")
        label.setGeometry(60, 16, 100, 30)

        self.contrastSlider = QSlider(Qt.Horizontal, self.contrastDialog)
        self.contrastSlider.setFocusPolicy(Qt.NoFocus)
        self.contrastSlider.setTickPosition(QSlider.TicksAbove)
        self.contrastSlider.setTickInterval(10)
        self.contrastSlider.setSingleStep(10)
        self.contrastSlider.setValue(50)
        self.contrastSlider.setGeometry(60, 51, 170, 50)
        self.contrastSlider.valueChanged.connect(self.setContrastFactor)

        self.contrastButton = QPushButton("OK", self.contrastDialog)
        self.contrastButton.move(110, 111)
        self.contrastButton.clicked.connect(self.setContrast)

        self.contrastDialog.exec_()

    def openOverlayImageChooser(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(self, "Choose an image", "",
                                                  "JPG (*.jpg);;PNG (*.png);;JPEG (*.jpeg);", options=options)
        fileName = ""
        if filePath:
            self.overlayImageFilePath = filePath
            fileName = filePath[filePath.rfind("/") + 1:]
            # hide the choose button
            self.overlayPanelImageButton.hide()
            self.overlayPanelImageLabel.setText(fileName)
            self.overlayPanelImageLabel.show()

    def openFileChooser(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(self, "Choose an image", "",
                                                  "JPG (*.jpg);;PNG (*.png);;JPEG (*.jpeg);", options=options)
        # fname, _ = QFileDialog.getOpenFileName(window, 'Open file', '.',
        #                                        'Image Files(*.jpg *.bmp *.png *.jpeg *.rgb *.tif)')
        if filePath:

            temp = cv2.imread(filePath)
            height, width, depth = temp.shape
            extension = ""
            if filePath.endswith(".jpg"):
                extension = "jpg"
            else:
                extension = "png"

            cv2.imwrite("originalImage." + extension, temp)

            fileSize = os.stat(filePath)
            fileSize = ((fileSize.st_size) / (1024 * 1024))
            fileSize = round(fileSize, 3)
            fileName = filePath[filePath.rfind("/") + 1:]

            self.images["undoStack"] = list()
            self.images["redoStack"] = list()

            self.images["undoStack"].append("originalImage." + extension)

            self.images["path"] = "originalImage." + extension
            self.images["name"] = fileName
            self.images["filterImageName"] = ""
            self.images["height"] = height
            self.images["width"] = width
            self.images["extension"] = extension
            self.images["size"] = fileSize
            self.images["status"] = "File opened..."
            self.update()
            self.createPanels()
            self.updateStatusBar()

    def saveFileChooser(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save", "", "JPEG (*.jpeg);;PNG (*.png)", options=options)
        if fileName:
            self.saveResult(fileName)

    def saveResult(self, saveWith):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        imagePath = self.images["filterImageName"]
        if imagePath == "":
            imagePath = self.images["path"]
        if imagePath.endswith(".png"):
            saveWith += ".png"
        else:
            saveWith += ".jpg"
        # 读取图片，将图片保存在指定路径
        print(f'save path {saveWith}')
        contents = cv2.imread(imagePath)
        cv2.imwrite(saveWith, contents)
        self.images["status"] = "file saved successfully..."
        self.updateStatusBar()

    def performFourierTransform(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(29)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performInverseFourierTransform(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(55)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performDiscreteCosineTransform(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(30)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performInverseDiscreteCosineTransform(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(56)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performDiscreteRadonTransform(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(31)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performGaussianNoiseImage(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(32)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performSaltPepperImage(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(33)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performHighPassFilter(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(36)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performMedianBlur(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(37)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performBlur(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(38)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performBilateralFilter(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(40)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performPixelFrequencyHistogram(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(42)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()


    def performGrayFrequencyHistogram(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(43)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performpSeudoColorEnhancement(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(45)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performTrueColorEnhancement(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(46)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performHistNormalized(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(47)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performBGR2NTSC(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(48)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performBGR2YCR_CB(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(49)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performBGR2HSV(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(50)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performBinalization(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(56)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performCannyEdgeDetection(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(57)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performOtsuBinarization(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(51)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performDilateImage(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(52)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performErodeImage(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(53)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performFeatureExtraction(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(54)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performClassification(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(55)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()

    def performOriginalImage(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        self.worker = Worker()
        self.worker.setOperation(58)
        self.worker.setTmApplication(self)
        self.worker.signal.connect(self.taskFinished)
        self.worker.start()
        self.showLoader()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    model = DataModel(1250, 700)
    window = TMApplication(model)
    splash = QSplashScreen()  # 显示一个启动画面。启动画面可用于显示产品信息、程序功能、广告等
    splash.setPixmap(QPixmap("img/splash-bg.png"))
    splash.show()
    statusLabel = QLabel(splash)
    # 从屏幕上（483,260）位置开始（即为最左上角的点），显示一个110*20的界面（宽110，高20）
    # 如果在控件中加上了layout布局，就会发现发现没有办法使用setGeometry函数了，这是因为布局已经被layout管理
    statusLabel.setGeometry(483, 260, 110, 20)
    statusLabel.show()
    # 提供了一个水平或垂直进度条
    progressBar = QProgressBar(splash)
    progressBar.setGeometry(3, 283, 592, 15)
    progressBar.setStyleSheet("QProgressBar::chunk{background-color:#9da603;}")
    # 对齐方式
    progressBar.setAlignment(QtCore.Qt.AlignCenter)
    progressBar.setFormat("Loading...")
    progressBar.show()
    j = 0
    logsSize = len(window.logFiles)
    for i in range(101):
        progressBar.setValue(i)
        if j < logsSize:
            time.sleep(0.5)
            statusLabel.setText(window.logFiles[i])
            j += 1
        t = time.time()
        while time.time() < t + 0.1:
            app.processEvents()
    splash.close()
    window.show()
    sys.exit(app.exec_())
