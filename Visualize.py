import numpy as np
import matplotlib.pyplot as plt

# takes two parameters which are paths of the .npy files which contain losses
# function merges two numpy arrays into a single one and returns it
def mergeLosses(loss1Path, loss2Path):
    # loads .npy files
    loss1 = np.load(loss1Path)
    loss2 = np.load(loss2Path)

    # inits final loss list
    lossList = loss1[0]
    # iterates through first loss list and combines it with the final list
    for i in range(1, len(loss1)):
        for j in range(len(loss1[i])):
            lossList = np.append(lossList, loss1[i][j])
    # iterates through second loss list and combines it with the final list
    for i in range(len(loss2)):
        for j in range(len(loss2[i])):
            lossList = np.append(lossList, loss2[i][j])
    # returns final loss list
    return lossList

# takes two parameters which are loss list and step size
# function takes average loss of batches with given step size
# creates a new list with the averages and plots it
# (not averaging causes very spiky loss plot which is hard to understand)
def visualizeLossPlot(lossList, step):
    # inits final loss list
    finalLoss = []
    finalLoss.append(lossList[0])
    sum = 0
    # iterates through each batch loss and takes averages of them (using given step size)
    for j in range(1, len(lossList)):
        sum = sum + lossList[j]
        if (j % step == 0):
            finalLoss.append(sum / step)
            sum = 0
    # plots final loss graph
    plt.plot(finalLoss)
    plt.show()


loss = mergeLosses("losses_1.npy", "losses_2.npy")
visualizeLossPlot(loss, 100)