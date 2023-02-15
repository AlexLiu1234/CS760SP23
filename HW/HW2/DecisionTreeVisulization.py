import math
from math import log
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import random


decisionNode = dict(boxstyle="sawtooth", fc='0.8')
leafNode = dict(boxstyle="round4", fc='0.8')
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_point, parent_point, node_style):
    plt.annotate(node_txt, xy=parent_point, xycoords="axes fraction", xytext=center_point,
                 textcoords='axes fraction', va='center', ha='center', bbox=node_style, arrowprops=arrow_args)


def getNumLeaves(my_tree):
    num_leaves = 0
    first_str = list(my_tree.keys())[0]
    next_dict = my_tree[first_str]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__ == 'dict':
            num_leaves += getNumLeaves(next_dict[key])
        else:
            num_leaves += 1
    return num_leaves
 

def getDepthTree(myTree):
    depthTree = 0
    firststr = list(myTree.keys())[0]
    nextDict = myTree[firststr]
    for key in nextDict.keys():
        if type(nextDict[key]).__name__ == 'dict':
            thisDepth = 1 + getDepthTree(nextDict[key])
        else:
            thisDepth = 1
        if thisDepth > depthTree:
            depthTree = thisDepth
    return depthTree


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeaves(myTree)
    depth = getDepthTree(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    creatPlot.ax1.text(xMid, yMid, nodeTxt)
    plot_node(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plot_node(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            xMid = (plotTree.xOff + cntrPt[0]) / 2.0
            yMid = (plotTree.yOff + cntrPt[1]) / 2.0
            creatPlot.ax1.text(xMid, yMid, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
 

def creatPlot(inTree):
    fig = plt.figure(1, facecolor='white', figsize=(10, 6))
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    creatPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeaves(inTree))
    plotTree.totalD = float(getDepthTree(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    dataSet, labels = LoadData('D2.txt')
    tree = BuildDecisionTree_FOR_TREEGRAPH(dataSet, labels)
    creatPlot(tree)