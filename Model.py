import numpy as np

from keras.applications.densenet import *
from keras.models import Model, Sequential
from keras.layers import Conv2D, UpSampling2D

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.training.adam import AdamOptimizer

from Dataset import DataHelper

from skimage import transform, color, io


class ColorizationModel:

    def __init__(self, path=None):
        """
        Construction method for Colorization Model
        """

        self.model = self.mergeModels(self.getLayersDenseNet())
        self.dataHelper = DataHelper()
        if path != None:
            self.model.load_weights(filepath=path)


    def getLayersDenseNet(self):
        """
        Collecting first 51 layer of pretrained densenet for feature extraction
        :return: layers of densenet
        """

        # changing input layer according our calculations
        # include_top false removes fully connected layer
        denseNet = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        for i in range(len(denseNet.layers) - 51):
            # removing layers
            denseNet.layers.pop()

        # we are copying densenet because of tensor flow does not allow us
        # change output layer directly
        croppedDenseNet = Model(denseNet.input, denseNet.layers[-1].output)
        return croppedDenseNet

    def mergeModels(self, denseNet):
        """
        Merging densenet and our layers
        :param denseNet:
        :return: colorization model
        """

        model = Sequential()
        model.add(denseNet)

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2,  (3, 3), activation='tanh', padding='same'))

        model.compile(optimizer='rmsprop', loss='mse')

        return model

    def train(self, trainDir, batchSize=32, epochs=1):
        """
        Training model
        :param trainDir: directory of training images
        :param batchSize: batch size
        :param epochs: number of epochs
        :return:
        """

        self.dataHelper.trainImageDir = trainDir
        self.dataHelper.batchSize = batchSize
        self.dataHelper.updateDir()

        cp_callback = ModelCheckpoint(
            "model_dir/model_1.cpkt", verbose=1, save_weights_only=True,
            # Save weights, every epoch.
            period=1)

        batchNumber = self.dataHelper.getNumberOfBatchesTrainSet()
        totalBatchNumber = batchNumber * epochs
        losses = []
        for epoch in range(epochs):
            print("EPOCH NUMBER:", epoch)

            epoch_losses = []
            for batchIndex in range(batchNumber):
                currentBatchInTotal = (batchIndex + 1) + (epoch * batchNumber)
                print("training in progress %{0:.2f}".format(float(currentBatchInTotal) / totalBatchNumber * 100))

                trainSamples, trainLabels = self.dataHelper.getBatchFromTrainSet(batchIndex)
                if batchIndex == batchNumber-1:
                    callback = self.model.fit(x=trainSamples, y=trainLabels, batch_size=batchSize, epochs=1,
                                   callbacks=[cp_callback], verbose=0)
                    hist = callback.history
                    epoch_losses.append(hist.get("loss")[0])
                    losses.append(epoch_losses)
                else:
                    callback = self.model.fit(x=trainSamples, y=trainLabels, batch_size=batchSize, epochs=1,
                                   verbose=0)
                    hist = callback.history
                    epoch_losses.append(hist.get("loss")[0])

        np.save("losses_1.npy", np.array(losses))

    def test(self, testDir):
        """
        Run forward propagation on model
        :param testDir: Test image directories
        :return:
        """
        self.dataHelper.testImageDir = testDir
        self.dataHelper.batchSize = 1
        self.dataHelper.updateDir()

        batchNumber = self.dataHelper.getNumberOfBatchesTestSet()

        for batchIndex in range(batchNumber-1):
            testSamples, testSamplesNames = self.dataHelper.getBatchFromTestSet(batchIndex)
            output = self.model.predict(x=testSamples)
            output = output * 128
            canvas = np.zeros((224, 224, 3))
            canvas[:, :, 0] = testSamples[0][:, :, 0]
            canvas[:, :, 1:] = output[0]
            output = color.lab2rgb(canvas)
            output = transform.resize(output, output_shape=(720, 1280))
            io.imsave("DataSets/colorized/" + testSamplesNames[0], output)

    def calculateSimilarityL2(self, ab0, ab1):
        """
        :param ab0:
        :param ab1:
        :return:
        """
        dist = 0.0
        for y in range(ab0.shape[0]):
            for x in range(ab0.shape[1]):
                diff = (ab0[y][x][0] - ab1[y][x][0])
                diff = 0 if diff < 5 else diff
                dist += diff

        return dist

    def calculateSimilarityPixelByPixel(self, ab0, ab1):
        """
        :param ab0:
        :param ab1:
        :return:
        """
        dist = 0.0
        for y in range(ab0.shape[0]):
            for x in range(ab0.shape[1]):
                diff = (ab0[y][x][0] - ab1[y][x][0])
                diff = 0 if diff < 5 else 1
                dist += diff

        return dist

    def evaluationResults(self, evaluationImagesDir):
        """
        This method evaluate results of each image in directory in evaluation image dir
        :param evaluationImages:
        :param resultImages:
        :return:
        """
        eval = open("DataSets/evaluation/distances.txt", "w")
        evalList = self.dataHelper.getEvaluationList(evaluationImagesDir)
        for i, imagePath in enumerate(evalList):
            print(i/len(evalList))
            image = io.imread(imagePath)

            image = transform.resize(image, (224, 224))
            image = color.rgb2lab(image)

            X = image[:, :, 0]
            X = np.stack((X,) * 3, axis=2)

            modelInput = np.empty(shape=(1, 224, 224, 3), dtype=np.uint8)
            modelInput[0] = X

            output = self.model.predict(modelInput)
            output = output[0]
            output *= 128

            outputImage = np.empty(shape=(224, 224, 3))
            outputImage[:, :, 0] = X[:, :, 0]
            outputImage[:, :, 1:] = output

            image_ab = image[:, :, 1:]

            distance = self.calculateSimilarityPixelByPixel(image_ab, output)

            io.imsave("DataSets/evaluation/" + imagePath.split("/")[2], color.lab2rgb(outputImage))
            eval.write(imagePath + "," + str(distance) + "\n")


def investigateEval(evalText):
    images = []
    distances = []
    file = open(evalText, "r")
    for line in file:
        line = line.split(",")
        images.append(line[0])
        distances.append(float(line[1]))

    for i in range(5):
        index = distances.index(min(distances))
        print(images[index], distances[index])

        distances.remove(min(distances))
        images.remove(images[index])


if __name__ == "__main__":
    model = ColorizationModel("finalModel.cpkt")
    model.test(testDir="videoFrames/VideoFrames")

