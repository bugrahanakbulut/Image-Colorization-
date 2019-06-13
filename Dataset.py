import numpy as np

from glob import glob

from math import ceil

from random import randint

import skimage.io as io
import skimage.color as color
import skimage.transform as transform

from skimage.util import random_noise


class DataHelper:
    def __init__(self, trainImageDir=None, testImageDir=None, batchSize=16):
        """
        :param trainImageDir:
        :param testImageDir:
        :param batchSize:
        """
        self.trainImageDir = trainImageDir if trainImageDir else None
        self.testImageDir = testImageDir if testImageDir else None
        self.batchSize = batchSize if batchSize else None

        self.trainImageNames = []
        self.testImageNames = []

    def updateDir(self):
        self.trainImageNames = glob(self.trainImageDir + "/*.jpg") if self.trainImageDir else None
        self.testImageNames = glob(self.testImageDir + "/*.jpg") if self.testImageDir else None

    def getNumberOfBatchesTrainSet(self):
        numberOfImages = len(self.trainImageNames)
        return ceil(numberOfImages/float(self.batchSize))

    def getNumberOfBatchesTestSet(self):
        numberOfImages = len(self.testImageNames)
        return ceil(numberOfImages/float(self.batchSize))

    def randomNoise(self, originalImage):
        return random_noise(originalImage)

    def rotateImageRandom(self, originalImage):
        return transform.rotate(originalImage, randint(-45, 45))

    def horizontalFlip(self, originalImage):
        return originalImage[:, ::-1]

    def verticalFlip(self, originalImage):
        return originalImage[::-1, :]

    def applyDataAugmentation(self, originalImage):
        images = []
        x = []
        y = []
        images.append(originalImage)
        images.append(self.randomNoise(originalImage))
        images.append(self.rotateImageRandom(originalImage))
        images.append(self.horizontalFlip(originalImage))
        images.append(self.verticalFlip(originalImage))

        for image in images:
            image = color.rgb2lab(image)

            X = image[:, :, 0]
            X = np.stack((X,) * 3, axis=2)

            Y = image[:, :, 1:]
            Y = Y / 128

            x.append(X)
            y.append(Y)

        return x, y

    def getBatchFromTrainSet(self, batchNumber):
        """
        Returns a batch from data set according to its' index
        :param batchNumber:
        :return:
        """
        if batchNumber > self.getNumberOfBatchesTrainSet():
            print("Batch Number requested bigger than total number of batches.")
            return -1

        batchStartinIndex = self.batchSize * batchNumber
        batchEndIndex = batchStartinIndex + self.batchSize
        batchEndIndex = batchEndIndex if batchEndIndex < len(self.trainImageNames) else len(self.trainImageNames) - 1
        grays = []
        abs = []
        for index in range(batchStartinIndex, batchEndIndex):
            image = io.imread(self.trainImageNames[index])
            image = transform.resize(image, (224, 224))
            image = color.rgb2lab(image)

            X = image[:, :, 0]
            X = np.stack((X,) * 3, axis=2)

            Y = image[:, :, 1:]
            Y = Y / 128

            grays.append(X)
            abs.append(Y)

        grays = self.listToNumpyArray(grays, (len(grays), 224, 224, 3))
        abs = self.listToNumpyArray(abs, (len(abs), 224, 224, 2))

        return grays, abs

    def getBatchFromTestSet(self, batchNumber=1):
        if batchNumber > self.getNumberOfBatchesTestSet():
            print("Batch Number requested bigger than total number of batches.")
            return -1

        batchStartingIndex = self.batchSize * batchNumber
        batchEndIndex = batchStartingIndex + self.batchSize
        batchEndIndex = batchEndIndex if batchEndIndex < len(self.testImageNames) else len(self.testImageNames) - 1

        testSamplesGray = []
        testSamplesName = []

        for index in range(batchStartingIndex, batchEndIndex):
            testSamplesName.append(self.testImageNames[index].split("/")[-1])
            image = io.imread(self.testImageNames[index])
            image = transform.resize(image, (224, 224))
            image = color.rgb2lab(image)

            X = image[:, :, 0]
            X = np.stack((X,) * 3, axis=2)

            testSamplesGray.append(X)

        testSamples = self.listToNumpyArray(testSamplesGray, (len(testSamplesGray),  224, 224, 3))
        return testSamples, testSamplesName

    def getEvaluationList(self, evalDir):
        return glob(evalDir + "/*.jpg")

    def listToNumpyArray(self, convList, shape):
        resultnp = np.empty(shape=shape)
        for index, item in enumerate(convList):
            resultnp[index] = item

        return resultnp

    def validateTrainSet(self):
        """
        :return:
        """
        self.updateDir()
        bnumber = self.getNumberOfBatchesTrainSet()
        for i in range(bnumber):
            self.getBatchFromTrainSet(i)



if __name__ == "__main__":
    dh = DataHelper(trainImageDir="DataSets/train_01", batchSize=32)
    dh.validateTrainSet()
    # dh.getBatchFromTrainSet(40)
