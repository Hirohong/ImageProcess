import numpy
import sys
import cv2
import math
import torch
import numpy as np
import os.path
from torchvision import transforms
import torchvision.models as models
import skimage.io
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.transform import radon
from skimage.color import rgb2yiq
from PIL import Image, ImageFont, ImageDraw
# 图像处理操作
is_print = True


def convertToRGB(value):
    if is_print:
        print(f"当前方法名：{sys._getframe().f_code.co_name}")
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


class ImageProcessor:
    def __init__(self, TMApplication):
        if is_print:
            print(f'当前类名称：{self.__class__.__name__}')
        self.tmApplication = TMApplication

    def performReplicateBorderOperation(self, borderDetails):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]

        imageData = cv2.imread(filePath)
        thickness = borderDetails["thickness"]
        borderType = borderDetails["border-type"]

        if borderType == "replicate_standard":
            outputImageData = cv2.copyMakeBorder(imageData, thickness, thickness, thickness, thickness,
                                                 cv2.BORDER_REFLECT)
        else:
            top = thickness[0]
            bottom = thickness[1]
            left = thickness[2]
            right = thickness[3]
            outputImageData = cv2.copyMakeBorder(imageData, top, bottom, left, right, cv2.BORDER_REFLECT)

        imName = "border-replicate"
        status = "Border replicate applied successfully...."
        self.updateImages(filePath, imName, outputImageData, status)


    def performNormalBorderOperation(self, borderDetails):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]

        imageData = cv2.imread(filePath)
        thickness = borderDetails["thickness"]
        color = borderDetails["color"]
        borderType = borderDetails["border-type"]
        color = convertToRGB(color)
        color = list(color)
        color[0], color[2] = color[2], color[0]

        if borderType == "normal_standard":
            outputImageData = cv2.copyMakeBorder(imageData, thickness, thickness, thickness, thickness,
                                                 cv2.BORDER_CONSTANT, value=color)
        else:
            top = thickness[0]
            bottom = thickness[1]
            left = thickness[2]
            right = thickness[3]
            outputImageData = cv2.copyMakeBorder(imageData, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                 value=color)
        imName = "border"
        status = "Border applied successfully...."
        self.updateImages(filePath, imName, outputImageData, status)


    def performTextWatermark(self, text, textSize, textThickness, opacity, coordinates):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]

        imageData = cv2.imread(filePath)
        overlay = imageData.copy()
        outputImageData = imageData.copy()
        height, width = imageData.shape[:2]

        alpha = opacity / 100
        cv2.putText(overlay, text.format(alpha), coordinates, cv2.FONT_HERSHEY_PLAIN, textSize, (0, 0, 0),
                    thickness=textThickness)
        cv2.addWeighted(overlay, alpha, outputImageData, 1 - alpha, 0, outputImageData)

        imName = "watermark-text"
        status = "Watermark applied successfully...."
        self.updateImages(filePath, imName, outputImageData, status)


    def performImageWatermark(self, sourceImage, opacity, coordinates):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        scale = 1
        imageData = cv2.imread(filePath, -1)
        outputImageData = imageData.copy()
        opacity = opacity / 100
        watermarkImageData = cv2.imread(sourceImage, -1)
        watermarkImageData = cv2.resize(watermarkImageData, (0, 0), fx=scale, fy=scale)
        height, width, depth = watermarkImageData.shape
        rows, columns, _ = imageData.shape
        y, x = coordinates[0], coordinates[1]
        for i in range(height):
            for j in range(width):
                if x + i >= rows or y + j >= columns:
                    continue
                alpha = float(watermarkImageData[i][j][3] / 255.0)  # read the alpha channel
                imageData[x + i][y + j] = alpha * watermarkImageData[i][j][:3] + (1 - alpha) * imageData[x + i][
                    y + j]

        cv2.addWeighted(imageData, opacity, outputImageData, 1 - opacity, 0, outputImageData)

        imName = "watermark-image"
        status = "Watermark applied successfully...."
        self.updateImages(filePath, imName, outputImageData, status)


    def performZoomOut(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height, width, depth = imageData.shape
        try:
            outputImageData = cv2.resize(imageData, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        except:
            print("Failed to perform operation. Invalid Input !")

        imName = "zoom-out"
        status = "Zoomed-out Successfully...."
        self.updateImages(filePath, imName, outputImageData, status)

        self.tmApplication.images["height"] = outputImageData.shape[0]
        self.tmApplication.images["width"] = outputImageData.shape[1]


    def performZoomIn(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        outputImageData = cv2.resize(imageData, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        imName = "zoom-in"
        status = "Zoomed-in Successfully...."
        self.updateImages(filePath, imName, outputImageData, status)

        self.tmApplication.images["height"] = outputImageData.shape[0]
        self.tmApplication.images["width"] = outputImageData.shape[1]


    def performTextOperation(self, text, fontPath, fontColor, fontSize, x, y):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = Image.open(filePath)
        draw = ImageDraw.Draw(imageData)
        font = ImageFont.truetype(fontPath, int(fontSize))
        fontColor = "rgb" + str(convertToRGB(fontColor))
        coordinates = list()
        coordinates.append(x)
        coordinates.append(y)
        coordinates = tuple(coordinates)
        draw.text(coordinates, text, fill=fontColor, font=font)

        # imageData.show()  # 添加文字成功

        # imName = "text"
        # status = "Text Applied Successfully...."
        # self.updateImages(filePath, imName, imageData, status)

        if filePath.endswith(".png"):
            imageData.save("text.png")
            filePath = "text.png"
        else:
            imageData.save("text.jpg")
            filePath = "text.jpg"

        fileSize = os.stat(filePath)
        fileSize = ((fileSize.st_size) / (1024 * 1024))
        fileSize = round(fileSize, 3)
        if filePath.endswith(".jpg"):
            self.tmApplication.images["extension"] = "jpg"
        else:
            self.tmApplication.images["extension"] = "png"
        self.tmApplication.images["size"] = fileSize
        self.tmApplication.images["status"] = "Text Applied Successfully...."
        self.tmApplication.images["undoStack"].append(filePath)
        self.tmApplication.images["filterImageName"] = filePath

    def performLaplace(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height, width, depth = imageData.shape

        grayData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        laplaceData = grayData
        verticalFilter = [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]

        horizontalFilter = [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]

        height = height - 3 + 1
        width = width - 3 + 1

        for i in range(height):
            for j in range(width):
                sumX = 0
                sumY = 0
                for k in range(3):
                    for l in range(3):
                        sumX += (verticalFilter[k][l] * grayData[i + k][j + l])
                        sumY += (horizontalFilter[k][l] * grayData[i + k][j + l])

                laplaceData[i][j] = math.sqrt(pow(sumX, 2) + pow(sumY, 2)) / math.sqrt(pow(1.9, 2) + pow(1.9, 2))

        imName = "laplace"
        status = "Converted to Laplace...."
        self.updateImages(filePath, imName, laplaceData, status)


    def performUnsharpening(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height, width, depth = imageData.shape

        kernel = [[1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9]]

        amount = 2
        offset = len(kernel) // 2

        outputImageData = np.zeros((height, width, depth))

        # Compute convolution with kernel
        for x in range(offset, height - offset):
            for y in range(offset, width - offset):
                original_pixel = imageData[x][y]
                acc = [0, 0, 0]
                for a in range(len(kernel)):
                    for b in range(len(kernel)):
                        xn = x + a - offset
                        yn = y + b - offset
                        pixel = imageData[xn][yn]
                        acc[0] += pixel[0] * kernel[a][b]
                        acc[1] += pixel[1] * kernel[a][b]
                        acc[2] += pixel[2] * kernel[a][b]

                new_pixel = (
                    int(original_pixel[0] + (original_pixel[0] - acc[0]) * amount),
                    int(original_pixel[1] + (original_pixel[1] - acc[1]) * amount),
                    int(original_pixel[2] + (original_pixel[2] - acc[2]) * amount)
                )
                outputImageData[x][y] = new_pixel

        imName = "unsharp"
        status = "Unsharpening Done...."
        self.updateImages(filePath, imName, outputImageData, status)


    def performSharpening(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        kernel = [[0, -.5, 0],
                  [-.5, 3, -.5],
                  [0, -.5, 0]]

        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height, width, depth = imageData.shape

        offset = len(kernel) // 2
        outputImageData = np.zeros(imageData.shape)

        for x in range(offset, height - offset):
            for y in range(offset, width - offset):
                acc = [0, 0, 0]
                for a in range(len(kernel)):
                    for b in range(len(kernel)):
                        xn = x + a - offset
                        yn = y + b - offset
                        pixel = imageData[xn][yn]
                        acc[0] += pixel[0] * kernel[a][b]
                        acc[1] += pixel[1] * kernel[a][b]
                        acc[2] += pixel[2] * kernel[a][b]

                outputImageData[x][y][0] = acc[0]
                outputImageData[x][y][1] = acc[1]
                outputImageData[x][y][2] = acc[2]

        imName = "sharpen"
        status = "Sharpening Done...."
        self.updateImages(filePath, imName, outputImageData, status)

    def performInvert(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height, width, depth = imageData.shape
        outputImageData = imageData
        for i in range(height):
            for j in range(width):
                blue = outputImageData[i][j][0]
                green = outputImageData[i][j][1]
                red = outputImageData[i][j][2]

                outputImageData[i][j][0] = 255 - blue
                outputImageData[i][j][1] = 255 - green
                outputImageData[i][j][2] = 255 - red

        imName = "invert"
        status = "Inverted...."
        self.updateImages(filePath, imName, outputImageData, status)

    def performBoxBlur(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height, width, depth = imageData.shape

        boxKernel = [
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ]

        offset = len(boxKernel) // 2
        outputImage = np.zeros((height, width, depth))

        for i in range(1):
            for x in range(offset, height - offset):
                for y in range(offset, width - offset):
                    red = 0
                    green = 0
                    blue = 0
                    for a in range(len(boxKernel)):
                        for b in range(len(boxKernel)):
                            xn = x + a - offset
                            yn = y + b - offset
                            pixels = imageData[xn][yn]
                            blue += pixels[0] * boxKernel[a][b]
                            green += pixels[1] * boxKernel[a][b]
                            red += pixels[2] * boxKernel[a][b]
                    outputImage[x][y][0] = blue
                    outputImage[x][y][1] = green
                    outputImage[x][y][2] = red

        imName = "boxBlur"
        status = "Box Blur applied successfully...."
        self.updateImages(filePath, imName, outputImage, status)


    def convolution(self, oldimage, kernel):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # image = Image.fromarray(image, 'RGB')
        image_h = oldimage.shape[0]
        image_w = oldimage.shape[1]

        kernel_h = kernel.shape[0]
        kernel_w = kernel.shape[1]

        if (len(oldimage.shape) == 3):
            image_pad = np.pad(oldimage,
                               pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2), (0, 0)),
                               mode='constant', constant_values=0).astype(np.float32)
        elif (len(oldimage.shape) == 2):
            image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2)),
                               mode='constant', constant_values=0).astype(np.float32)

        h = kernel_h // 2
        w = kernel_w // 2

        image_conv = np.zeros(image_pad.shape)

        for i in range(h, image_pad.shape[0] - h):
            for j in range(w, image_pad.shape[1] - w):
                # sum = 0
                x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
                x = x.flatten() * kernel.flatten()
                image_conv[i][j] = x.sum()
        h_end = -h
        w_end = -w

        if (h == 0):
            return image_conv[h:, w:w_end]
        if (w == 0):
            return image_conv[h:h_end, w:]
        return image_conv[h:h_end, w:w_end]

    def performHorizontalMotionBlur(self, kernelSize):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        image = cv2.imread(filePath)
        horizontalKernel = np.zeros((kernelSize, kernelSize))
        horizontalKernel[int((kernelSize - 1) / 2), :] = np.ones(kernelSize)
        for i in range(kernelSize):
            for j in range(kernelSize):
                horizontalKernel[i][j] /= kernelSize
        im_filtered = np.zeros_like(image)
        for c in range(3):
            im_filtered[:, :, c] = self.convolution(image[:, :, c], horizontalKernel)
        blurData = (im_filtered.astype(np.uint8))

        imName = "horizontalMotion-blur"
        status = "Horizontal Motion Blur applied successfully...."
        self.updateImages(filePath, imName, blurData, status)

    def performVerticalMotionBlur(self, kernelSize):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        image = cv2.imread(filePath)

        verticalKernel = np.zeros((kernelSize, kernelSize))
        verticalKernel[:, int((kernelSize - 1) / 2)] = np.ones(kernelSize)
        for i in range(kernelSize):
            for j in range(kernelSize):
                verticalKernel[i][j] /= kernelSize
        im_filtered = np.zeros_like(image)
        for c in range(3):
            im_filtered[:, :, c] = self.convolution(image[:, :, c], verticalKernel)
        blurData = (im_filtered.astype(np.uint8))

        imName = "verticalMotion-blur"
        status = "Vertical Motion Blur applied successfully...."
        self.updateImages(filePath, imName, blurData, status)

    def performGuassianBlur(self, radius):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        image = cv2.imread(filePath)

        filter_size = 2 * int(4 * radius + 0.5) + 1

        result = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
        blurData = result.astype(np.uint8)

        imName = "guassian-blur"
        status = "Guassian blur applied successfully...."
        self.updateImages(filePath, imName, blurData, status)

    def performResize(self, destHeight, destWidth):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        sourceImage = cv2.imread(filePath)
        sourceHeight = sourceImage.shape[0]
        sourceWidth = sourceImage.shape[1]

        destImage = numpy.zeros((destHeight, destWidth, 3))

        srcY = (sourceHeight / destHeight)
        srcX = (sourceWidth / destWidth)

        for i in range(destHeight):
            for j in range(destWidth):
                dstX = math.ceil(j * srcX)
                dstY = math.ceil(i * srcY)
                if dstX == sourceWidth or dstY == sourceHeight: break
                destImage[i][j] = sourceImage[dstY][dstX]

        imName = "resize"
        status = "Resized successfully...."
        self.updateImages(filePath, imName, destImage, status)

        self.tmApplication.images["height"] = destHeight
        self.tmApplication.images["width"] = destWidth


    def performOverlay(self, startingX, startingY):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageOne = cv2.imread(filePath)
        imageTwo = cv2.imread(self.tmApplication.overlayImageFilePath)

        height = imageTwo.shape[0]
        width = imageTwo.shape[1]

        r = int(startingX)
        rr = 0
        while rr < height and r < imageOne.shape[0]:
            c = int(startingY)
            cc = 0
            while cc < width and c < imageOne.shape[1]:
                imageOne[r][c] = imageTwo[rr][cc]
                c += 1
                cc += 1
            r += 1
            rr += 1

        imName = "overlay"
        status = "Overlay done successfully...."
        self.updateImages(filePath, imName, imageOne, status)

    def performClockwise(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height = imageData.shape[0]
        width = imageData.shape[1]
        newImageData = numpy.zeros((width, height, 3))

        r = 0
        rr = height - 1
        while r < height:
            c = 0
            cc = 0
            while c < width:
                newImageData[cc][rr] = imageData[r][c]
                c += 1
                cc += 1
            r += 1
            rr -= 1

        imName = "cw-90"
        status = "Clockwise successfully..."
        self.updateImages(filePath, imName, newImageData, status)

        self.tmApplication.images["height"] = height
        self.tmApplication.images["width"] = width


    def performAnticlockwise(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height = imageData.shape[0]
        width = imageData.shape[1]

        newImageData = numpy.zeros((width, height, 3))

        r = 0
        rr = 0
        while r < height:
            c = width - 1
            cc = 0
            while c > 0:
                newImageData[cc][rr] = imageData[r][c]
                c -= 1
                cc += 1
            r += 1
            rr += 1

        imName = "acw-90"
        status = "Anticlockwise successfully..."
        self.updateImages(filePath, imName, newImageData, status)

        self.tmApplication.images["height"] = height
        self.tmApplication.images["width"] = width

    def perform180(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height = imageData.shape[0]
        width = imageData.shape[1]

        newImageData = numpy.zeros((height, width, 3))

        r = 0
        rr = height - 1
        while r < height:
            c = 0
            cc = width - 1
            while c < width:
                newImageData[rr][cc] = imageData[r][c]
                c += 1
                cc -= 1
            r += 1
            rr -= 1

        imName = "rotate-180"
        status = "Rotated successfully..."
        self.updateImages(filePath, imName, newImageData, status)

        self.tmApplication.images["height"] = height
        self.tmApplication.images["width"] = width


    def performHorizontalFlip(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height = imageData.shape[0]
        width = imageData.shape[1]

        newImageData = numpy.zeros((height, width, 3))

        r = 0
        rr = 0
        while r < height:
            c = 0
            cc = width - 1
            while c < width:
                newImageData[rr][cc] = imageData[r][c]
                c += 1
                cc -= 1
            r += 1
            rr += 1

        imName = "flip-Horizontal"
        status = "Flipped Horizontally..."
        self.updateImages(filePath, imName, newImageData, status)

    def performVerticalFlip(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height = imageData.shape[0]
        width = imageData.shape[1]

        newImageData = numpy.zeros((height, width, 3))

        r = 0
        rr = height - 1
        while r < height:
            c = 0
            cc = 0
            while c < width:
                newImageData[rr][cc] = imageData[r][c]
                c += 1
                cc += 1
            r += 1
            rr -= 1

        imName = "flip-vertical"
        status = "Flipped Vertically..."
        self.updateImages(filePath, imName, newImageData, status)

    def performCrop(self, startingX, startingY, endingX, endingY):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        cropSize = (endingX, endingY)
        c1 = int(startingX)
        r1 = int(startingY)
        c2 = int(endingY)
        r2 = int(endingX)
        if r2 >= imageData.shape[0]: r2 = imageData.shape[0] - 1
        if c2 >= imageData.shape[1]: c2 = imageData.shape[1] - 1
        cropSize = (c2 - c1 + 1, r2 - r1 + 1)
        newImage = numpy.zeros((cropSize[1], cropSize[0], 3))
        rr = 0
        r = r1
        while r <= r2:
            cc = 0
            c = c1
            while c <= c2:
                newImage[rr][cc] = imageData[r][c]
                c += 1
                cc += 1
            r += 1
            rr += 1

        imName = "crop"
        status = "Cropped Successfully..."
        self.updateImages(filePath, imName, newImage, status)

        self.tmApplication.images["height"] = endingY
        self.tmApplication.images["width"] = endingX


    def getPreviewForCrop(self, startingX, startingY, endingX, endingY):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        cropSize = (endingX, endingY)
        c1 = int(startingX)
        r1 = int(startingY)
        c2 = int(endingY)
        r2 = int(endingX)
        if r2 >= imageData.shape[0]: r2 = imageData.shape[0] - 1
        if c2 >= imageData.shape[1]: c2 = imageData.shape[1] - 1
        cropSize = (c2 - c1 + 1, r2 - r1 + 1)

        r = r1
        while r <= r2:
            imageData[r][c1] = (0, 0, 255)
            imageData[r][c2] = (0, 0, 255)
            r += 1
        c = c1
        while c <= c2:
            imageData[r1][c] = (0, 0, 255)
            imageData[r2][c] = (0, 0, 255)
            c += 1

        imName = "preview"
        status =  "Showing Preview..."
        self.updateImages(filePath, imName, imageData, status)


    def performContrastOperation(self, contrastFactor):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height, width, depth = imageData.shape
        grayData = np.zeros((height, width, depth))
        f = (259 * (contrastFactor + 255)) / (255 * (259 - contrastFactor))
        howMuch = (255 / 100) * contrastFactor
        for i in range(height):
            for j in range(width):
                rgb = imageData[i][j]
                red = rgb[0]
                green = rgb[1]
                blue = rgb[2]
                newRed = (f * (red - 128)) + 128
                newGreen = (f * (green - 128)) + 128
                newBlue = (f * (blue - 128)) + 128
                if contrastFactor < 50:
                    newRed -= howMuch
                    newGreen -= howMuch
                    newBlue -= howMuch
                else:
                    newRed += howMuch
                    newGreen += howMuch
                    newBlue += howMuch
                if newRed > 255: newRed = 255
                if newRed < 0: newRed = 0
                if newGreen > 255: newGreen = 255
                if newGreen < 0: newGreen = 0
                if newBlue > 255: newBlue = 255
                if newBlue < 0: newBlue = 0
                grayData[i][j] = (newRed, newGreen, newBlue)

        imName = "contrast"
        status = "Done..."
        self.updateImages(filePath, imName, grayData, status)

    def performBrightnessOperation(self, value):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height, width, depth = imageData.shape
        grayData = np.zeros((height, width, depth))
        brightnessFactor = 100 - value
        x = int((255 / 100) * brightnessFactor)
        for i in range(height):
            for j in range(width):
                rgb = imageData[i][j]
                red = rgb[0]
                green = rgb[1]
                blue = rgb[2]
                if brightnessFactor < 50:
                    red += x
                    green += x
                    blue += x
                else:
                    red -= x
                    green -= x
                    blue -= x
                if red > 255: red = 255
                if red < 0: red = 0
                if green > 255: green = 255
                if green < 0: green = 0
                if blue > 255: blue = 255
                if blue < 0: blue = 0
                grayData[i][j] = (red, green, blue)

        imName = "brightness"
        status = "Done..."
        self.updateImages(filePath, imName, grayData, status)

    def performGrayScale(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # filePath = self.tmApplication.images["path"]
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        height, width, depth = imageData.shape

        grayData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        grayData = np.stack((grayData,) * 3, axis=-1)

        imName = "grayScale"
        status = "Converted to Gray Scale successfully..."
        self.updateImages(filePath, imName, grayData, status)

    def updateImages(self, filePath, imName, outputImageData, status, imsave=False):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")

        if filePath.endswith(".png"):
            filePath = imName + ".png"
        else:
            filePath = imName + ".jpg"

        if imsave:
            skimage.io.imsave(filePath, outputImageData)
        else:
            cv2.imwrite(filePath, outputImageData)
        fileSize = os.stat(filePath)
        fileSize = ((fileSize.st_size) / (1024 * 1024))
        fileSize = round(fileSize, 3)
        if filePath.endswith(".jpg"):
            self.tmApplication.images["extension"] = "jpg"
        else:
            self.tmApplication.images["extension"] = "png"
        self.tmApplication.images["size"] = fileSize
        self.tmApplication.images["status"] = status
        self.tmApplication.images["undoStack"].append(filePath)
        self.tmApplication.images["filterImageName"] = filePath

    # 变换按钮相关方法
    # 傅里叶变换
    def fourierTransform(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        b, g, r = cv2.split(imageData)
        b_freImg = self.oneChannelDft(b)
        g_freImg = self.oneChannelDft(g)
        r_freImg = self.oneChannelDft(r)
        outputImageData = cv2.merge([b_freImg, g_freImg, r_freImg])

        imName = "fourierTransform"
        status = "Fourier Transform successfully..."
        self.updateImages(filePath, imName, outputImageData, status)

    # 傅里叶逆变换
    def inverseFourierTransform(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        b, g, r = cv2.split(imageData)
        b_recImg = self.oneChannelDft(b, dft=False)
        g_recImg = self.oneChannelDft(g, dft=False)
        r_recImg = self.oneChannelDft(r, dft=False)
        outputImageData = cv2.merge([b_recImg, g_recImg, r_recImg])

        imName = "inversefourierTransform"
        status = "inverse Fourier Transform successfully..."
        self.updateImages(filePath, imName, outputImageData, status)

    def oneChannelDft(self, img, dft=True):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")

        width, height = img.shape
        nwidth = cv2.getOptimalDFTSize(width)
        nheight = cv2.getOptimalDFTSize(height)
        # print(nwidth, nheigth)
        nimg = np.zeros((nwidth, nheight))
        nimg[:width, :height] = img
        # 傅里叶变换
        if dft:
            dft = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
            ndft = dft[:width, :height]
            ndshift = np.fft.fftshift(ndft)
            magnitude = np.log(cv2.magnitude(ndshift[:, :, 0], ndshift[:, :, 1]))  # 频谱图
            result = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255
            # frequencyImg = result.astype('uint8')
            result = result.astype('uint8')
        # 傅里叶逆变换
        else:
            ilmg = cv2.idft(dft)
            ilmg = cv2.magnitude(ilmg[:, :, 0], ilmg[:, :, 1])[:width, :height]
            ilmg = np.floor((ilmg - ilmg.min()) / (ilmg.max() - ilmg.min()) * 255)
            # recoveredImg = ilmg.astype('uint8')
            result = ilmg.astype('uint8')
        return result

    # 离散余弦变换
    def discreteCosineTransform(self):

        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        b, g, r = cv2.split(imageData)
        b_freImg = self.oneChannelDct(b)
        g_freImg = self.oneChannelDct(g)
        r_freImg = self.oneChannelDct(r)
        outputImageData = cv2.merge([b_freImg, g_freImg, r_freImg])

        imName = "discreteCosineTransform"
        status = "Discrete Cosine Transform successfully..."
        self.updateImages(filePath, imName, outputImageData, status)

    # 离散余弦反变换
    def inverseDiscreteCosineTransform(self):

        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        b, g, r = cv2.split(imageData)
        b_recImg = self.oneChannelDct(b)
        g_recImg = self.oneChannelDct(g)
        r_recImg = self.oneChannelDct(r)
        outputImageData = cv2.merge([b_recImg, g_recImg, r_recImg])

        imName = "inverseDiscreteCosineTransform"
        status = "Inverse Discrete Cosine Transform successfully..."
        self.updateImages(filePath, imName, outputImageData, status)

    def oneChannelDct(self, img, dct=True):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        # 离散余弦变换
        if dct:
            dct = cv2.dct(np.float32(img))
            # dct = np.log(abs(dct))
            result = dct.astype('uint8')
            # cv2.imshow("dct", frequencyImg)
        # 离散余弦反变换
        else:
            ilmg = cv2.idct(dct)
            result = ilmg.astype('uint8')
            # cv2.imshow("ilmg", recoveredImg)
        return result

    # Radon变换
    def discreteRadonTransform(self):

        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        img1 = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)

        theta = np.linspace(0., 180., max(img1.shape), endpoint=False)
        sinogram = radon(img1, theta=theta, circle=True)

        sinogram = sinogram.astype('uint8')
        outputImageData = np.stack((sinogram,) * 3, axis=-1)

        imName = "discreteRadonTransform"
        status = "Discrete Radon Transform successfully..."
        self.updateImages(filePath, imName, outputImageData, status)

    # 噪声按钮相关方法
    # 高斯噪声
    def addGaussianNoise(self, image, percetage):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        G_Noiseimg = image
        G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
        for i in range(G_NoiseNum):
            temp_x = np.random.randint(0, image.shape[0])  # (20, 40)
            temp_y = np.random.randint(0, image.shape[1])
            G_Noiseimg[temp_x][temp_y] = 255
        return G_Noiseimg

    def gaussianNoiseImage(self):

        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        # grayImage = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)  # 灰度变换
        outputImageData = self.addGaussianNoise(imageData, 0.02)  # 添加10%的高斯噪声 0.01
        # result = cv2.GaussianBlur(img[0], (3, 3), 0)
        imName = "addGaussianNoise"
        status = "Add Gaussian Noise successfully..."
        self.updateImages(filePath, imName, outputImageData, status)

    # 椒盐噪声
    def saltpepper(self, img, n):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")

        m = int((img.shape[0] * img.shape[1]) * n)
        for a in range(m):
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            if img.ndim == 2:
                img[j, i] = 255
            elif img.ndim == 3:
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
        for b in range(m):
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            if img.ndim == 2:
                img[j, i] = 0
            elif img.ndim == 3:
                img[j, i, 0] = 0
                img[j, i, 1] = 0
                img[j, i, 2] = 0
        return img

    def saltPepperImage(self):

        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        outputImageData = self.saltpepper(imageData, 0.02)

        imName = "addSaltPepperNoise"
        status = "Add Salt Pepper Noise successfully..."
        self.updateImages(filePath, imName, outputImageData, status)


    # 滤波按钮相关方法
    # 高通滤波
    def highPassFilter(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        x = cv2.Sobel(imageData, cv2.CV_16S, 1, 0)  # sobel算子进行梯度计算
        y = cv2.Sobel(imageData, cv2.CV_16S, 0, 1)
        absx = cv2.convertScaleAbs(x)  # 实现将原图片转换为uint8类型
        absy = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)  # 实现以不同的权重将两幅图片叠加

        imName = "highPassFilter"
        status = "high Pass Filter successfully..."
        self.updateImages(filePath, imName, result, status)

    # 低通滤波
    def medianBlur(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)

        # result = cv2.medianBlur(imageData, 5)  # 中值滤波(非线性平滑滤波器)，5当前方框的尺寸
        f = np.fft.fft2(imageData)
        fshift = np.fft.fftshift(f)
        # 2.制作掩膜图像，通低频
        rows, cols = imageData.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
        fshift = mask * fshift
        # 3.反傅里叶变换
        ishift = np.fft.ifftshift(fshift)
        result = np.abs(np.fft.ifft2(ishift))

        # # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
        # radius = 10
        # fft = cv2.dft(np.float32(imageData), flags=cv2.DFT_COMPLEX_OUTPUT)
        # # 对fft进行中心化，生成的dshift仍然是一个三维数组
        # dshift = np.fft.fftshift(fft)
        #
        # # 得到中心像素
        # rows, cols = imageData.shape[:2]
        # mid_row, mid_col = int(rows / 2), int(cols / 2)
        #
        # # 构建掩模，256位，两个通道
        # mask = np.zeros((rows, cols, 2), np.float32)
        # mask[mid_row - radius:mid_row + radius, mid_col - radius:mid_col + radius] = 1
        #
        # # 给傅里叶变换结果乘掩模
        # fft_filtering = dshift * mask
        # # 傅里叶逆变换
        # ishift = np.fft.ifftshift(fft_filtering)
        # image_filtering = cv2.idft(ishift)
        # image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
        # # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
        # result = cv2.normalize(image_filtering, None, 0, 1, cv2.NORM_MINMAX)

        result = np.stack((result,) * 3, axis=-1)
        print(f'result.shape {result.shape}')
        print(f'result {result}')

        imName = "medianBlur"
        status = "Median Blur successfully..."
        self.updateImages(filePath, imName, result, status)

    # 平滑滤波
    def blur(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        result = cv2.blur(imageData, (5, 5))  # 均值滤波5*5(线性平滑滤波器)

        imName = "medianBlur"
        status = "Median Blur successfully..."
        self.updateImages(filePath, imName, result, status)

    # 锐化滤波
    def bilateralFilter(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        # result = cv2.bilateralFilter(imageData, 9, 75, 75)  # 双边滤波，9领域直径，75空间高斯函数标准差，75灰度值相似性高斯函数标准差

        kernel = numpy.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ])
        result = cv2.filter2D(imageData, -1, kernel)

        imName = "bilateralFilter"
        status = "Bilateral Filter Blur successfully..."
        self.updateImages(filePath, imName, result, status)

    # 直方图统计按钮相关方法
    # 像素频数直方图
    def pixelFrequencyHistogram(self):

        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([imageData], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.savefig("hist1.jpg")
        result = cv2.imread("hist1.jpg")

        imName = "pixelFrequencyHistogram"
        status = "Pixel Frequency Histogram successfully..."
        self.updateImages(filePath, imName, result, status)


    # 灰度频数直方图
    def grayFrequencyHistogram(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        img1 = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        plt.hist(img1.ravel(), 256)
        plt.savefig("hist2.jpg")

        result = cv2.imread("hist2.jpg")

        imName = "grayFrequencyHistogram"
        status = "Gray Frequency Histogram successfully..."
        self.updateImages(filePath, imName, result, status)

    # 图像增强按钮相关方法
    # 伪彩色增强
    def pseudoColorEnhancement(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        grayImage = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)  # 灰度变换
        result = cv2.applyColorMap(grayImage, cv2.COLORMAP_JET)

        imName = "pseudoColorEnhancement"
        status = "Pseudo Color Enhancement successfully..."
        self.updateImages(filePath, imName, result, status)

    # 真彩色增强
    def trueColorEnhancement(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        Cimg = imageData / 255
        # 伽玛变换
        gamma = 0.5
        result = np.power(Cimg, gamma)
        result = result * 255

        imName = "trueColorEnhancement"
        status = "True Color Enhancement successfully..."
        self.updateImages(filePath, imName, result, status)

    # 直方图均衡
    def histNormalized(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        # imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        result = self.hist_equal(imageData)
        # result = np.stack((result,) * 3, axis=-1)

        imName = "histNormalized"
        status = "Hist Normalized successfully..."
        self.updateImages(filePath, imName, result, status)


    def hist_equal(self, img, z_max=255):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")

        H, W, C = img.shape
        S = H * W * C * 1.

        out = img.copy()

        sum_h = 0.

        for i in range(1, 255):
            ind = np.where(img == i)
            sum_h += len(img[ind])
            z_prime = z_max / S * sum_h
            out[ind] = z_prime

        out = out.astype(np.uint8)

        return out

    # NTSC颜色模型
    def bGR2NTSC(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2RGB)
        result = rgb2yiq(imageData)

        imName = "bGR2NTSC"
        status = "BGR2NTSC successfully..."
        self.updateImages(filePath, imName, result, status, imsave=True)

    # YCbCr颜色模型
    def bGR2YCR_CB(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        result = cv2.cvtColor(imageData, cv2.COLOR_BGR2YCR_CB)

        imName = "bGR2YCR_CB"
        status = "BGR2YCR_CB successfully..."
        self.updateImages(filePath, imName, result, status)

    # HSV颜色模型
    def bGR2HSV(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        result = cv2.cvtColor(imageData, cv2.COLOR_BGR2HSV)

        imName = "bGR2YCR_CB"
        status = "BGR2YCR_CB successfully..."
        self.updateImages(filePath, imName, result, status)

    # 图像分割按钮相关方法
    # 二值阈值分割
    def binalization(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        grayImage = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)  # 灰度变换
        _, result = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        result = np.stack((result,) * 3, axis=-1)

        imName = "binalization"
        status = "Binalization successfully..."
        self.updateImages(filePath, imName, result, status)

    # Canny边缘检测
    def cannyEdgeDetection(self, threshold1=32, threshold2=64):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        result = cv2.Canny(imageData, int(threshold1), int(threshold2))
        print(f'threshold1 {threshold1}')
        print(f'threshold2 {threshold2}')
        imName = "cannyEdgeDetection"
        status = "Canny Edge Detection successfully..."
        self.updateImages(filePath, imName, result, status)

    # 滞后阈值
    def Canny(self, img):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")

        # Gaussian filter for grayscale
        def gaussian_filter(img, K_size=3, sigma=1.3):

            if len(img.shape) == 3:
                H, W, C = img.shape
                gray = False
            else:
                img = np.expand_dims(img, axis=-1)
                H, W, C = img.shape
                gray = True

            # Zero padding
            pad = K_size // 2
            out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=float)
            out[pad: pad + H, pad: pad + W] = img.copy().astype(float)

            # prepare Kernel
            K = np.zeros((K_size, K_size), dtype=float)
            for x in range(-pad, -pad + K_size):
                for y in range(-pad, -pad + K_size):
                    K[y + pad, x + pad] = np.exp(- (x ** 2 + y ** 2) / (2 * sigma * sigma))
            # K /= (sigma * np.sqrt(2 * np.pi))
            K /= (2 * np.pi * sigma * sigma)
            K /= K.sum()

            tmp = out.copy()

            # filtering
            for y in range(H):
                for x in range(W):
                    for c in range(C):
                        out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

            out = np.clip(out, 0, 255)
            out = out[pad: pad + H, pad: pad + W]
            out = out.astype(np.uint8)

            if gray:
                out = out[..., 0]

            return out

        # sobel filter
        def sobel_filter(img, K_size=3):
            if len(img.shape) == 3:
                H, W, C = img.shape
            else:
                H, W = img.shape

            # Zero padding
            pad = K_size // 2
            out = np.zeros((H + pad * 2, W + pad * 2), dtype=float)
            out[pad: pad + H, pad: pad + W] = img.copy().astype(float)
            tmp = out.copy()

            out_v = out.copy()
            out_h = out.copy()

            # Sobel vertical
            Kv = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
            # Sobel horizontal
            Kh = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

            # filtering
            for y in range(H):
                for x in range(W):
                    out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
                    out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))

            out_v = np.clip(out_v, 0, 255)
            out_h = np.clip(out_h, 0, 255)

            out_v = out_v[pad: pad + H, pad: pad + W]
            out_v = out_v.astype(np.uint8)
            out_h = out_h[pad: pad + H, pad: pad + W]
            out_h = out_h.astype(np.uint8)

            return out_v, out_h

        def get_edge_angle(fx, fy):
            # get edge strength
            edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
            edge = np.clip(edge, 0, 255)

            fx = np.maximum(fx, 1e-10)
            # fx[np.abs(fx) <= 1e-5] = 1e-5

            # get edge angle
            angle = np.arctan(fy / fx)

            return edge, angle

        def angle_quantization(angle):
            angle = angle / np.pi * 180
            angle[angle < -22.5] = 180 + angle[angle < -22.5]
            _angle = np.zeros_like(angle, dtype=np.uint8)
            _angle[np.where(angle <= 22.5)] = 0
            _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
            _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
            _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

            return _angle

        def non_maximum_suppression(angle, edge):
            H, W = angle.shape
            _edge = edge.copy()

            for y in range(H):
                for x in range(W):
                    if angle[y, x] == 0:
                        dx1, dy1, dx2, dy2 = -1, 0, 1, 0
                    elif angle[y, x] == 45:
                        dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                    elif angle[y, x] == 90:
                        dx1, dy1, dx2, dy2 = 0, -1, 0, 1
                    elif angle[y, x] == 135:
                        dx1, dy1, dx2, dy2 = -1, -1, 1, 1
                    if x == 0:
                        dx1 = max(dx1, 0)
                        dx2 = max(dx2, 0)
                    if x == W - 1:
                        dx1 = min(dx1, 0)
                        dx2 = min(dx2, 0)
                    if y == 0:
                        dy1 = max(dy1, 0)
                        dy2 = max(dy2, 0)
                    if y == H - 1:
                        dy1 = min(dy1, 0)
                        dy2 = min(dy2, 0)
                    if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                        _edge[y, x] = 0

            return _edge

        def hysterisis(edge, HT=100, LT=30):
            H, W = edge.shape

            # Histeresis threshold
            edge[edge >= HT] = 255
            edge[edge <= LT] = 0

            _edge = np.zeros((H + 2, W + 2), dtype=np.float32)
            _edge[1: H + 1, 1: W + 1] = edge

            # 8 - Nearest neighbor
            nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

            for y in range(1, H + 2):
                for x in range(1, W + 2):
                    if _edge[y, x] < LT or _edge[y, x] > HT:
                        continue
                    if np.max(_edge[y - 1:y + 2, x - 1:x + 2] * nn) >= HT:
                        _edge[y, x] = 255
                    else:
                        _edge[y, x] = 0

            edge = _edge[1:H + 1, 1:W + 1]

            return edge

        # grayscale
        # gray = self.BGR2GRAY(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # gaussian filtering
        gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)

        # sobel filtering
        fy, fx = sobel_filter(gaussian, K_size=3)

        # get edge strength, angle
        edge, angle = get_edge_angle(fx, fy)

        # angle quantization
        angle = angle_quantization(angle)

        # non maximum suppression
        edge = non_maximum_suppression(angle, edge)

        # hysterisis threshold
        out = hysterisis(edge, 50, 20)

        return out

    # Gray scale
    def BGR2GRAY(img):
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        # Gray scale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        return out

    # Otsu Binalization
    def otsuBinarization(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        gray = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)

        thresh, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        result = np.stack((out,) * 3, axis=-1)

        imName = "otsuBinarization"
        status = "Otsu Binarization successfully..."
        self.updateImages(filePath, imName, result, status)


    # Morphology Erode
    def Morphology_Erode(self, img, Dil_time=1):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")

        H, W = img.shape

        # kernel
        MF = np.array(((0, 1, 0),
                       (1, 0, 1),
                       (0, 1, 0)), dtype=int)

        # each dilate time
        out = img.copy()
        for i in range(Dil_time):
            tmp = np.pad(out, (1, 1), 'edge')
            for y in range(1, H + 1):
                for x in range(1, W + 1):
                    if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) >= 255:
                        out[y - 1, x - 1] = 255

        return out


    # 形态学处理按钮相关方法
    # 膨胀
    def dilateImage(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        gray = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = cv2.dilate(gray, kernel, 10)
        result = np.stack((out,) * 3, axis=-1)

        imName = "dilateImage"
        status = "Dilate Image successfully..."
        self.updateImages(filePath, imName, result, status)

    # 腐蚀
    def erodeImage(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        gray = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = cv2.erode(gray, kernel, 10)
        result = np.stack((out,) * 3, axis=-1)

        imName = "erodeImage"
        status = "Erode Image successfully..."
        self.updateImages(filePath, imName, result, status)

    # 特征提取方法 角点检测
    def featureExtraction(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)

        gray = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        imageData[dst > 0.01 * dst.max()] = [0, 0, 255]

        imName = "featureExtraction"
        status = "Feature Extraction successfully..."
        self.updateImages(filePath, imName, imageData, status)

    # 图像分类与识别
    def classification(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["filterImageName"]
        if filePath == "": filePath = self.tmApplication.images["path"]
        imageData = cv2.imread(filePath)
        # img1 = img[0].copy()

        img = cv2.resize(imageData, (32, 32))
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4915, 0.4823, 0.4468],
                std=[1.0, 1.0, 1.0]
            )
        ])

        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        # pretrained=True就可以使用预训练的模型
        net = models.resnet50(pretrained=False)
        pthfile = r'resnet50.pth'
        net.load_state_dict(torch.load(pthfile))
        net.eval()

        outputs = net(img)
        confidence, classId = torch.max(outputs.data, 1)
        print(confidence, classId)

        # 加载类别
        classes = None
        with open("classification_classes_ILSVRC2012.txt", 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        # # 输出运行时间
        # t, _ = net.getPerfProfile()
        # label = 'cost time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        # print(label)
        # cv2.putText(imageData, label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 显示结果
        label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
        print(label)
        cv2.putText(imageData, label, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        imName = "classification"
        status = "Classification successfully..."
        self.updateImages(filePath, imName, imageData, status)

    def getOriginalImage(self):
        if is_print:
            print(f"当前方法名：{self.__class__.__name__}.{sys._getframe().f_code.co_name}")
        filePath = self.tmApplication.images["path"]
        if filePath == "": filePath = self.tmApplication.images["filterImageName"]
        imageData = cv2.imread(filePath)

        imName = "originalImage"
        status = "Get Original Image successfully..."
        self.updateImages(filePath, imName, imageData, status)