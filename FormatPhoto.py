import time

import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def rescaleImage(frame, scalar=0.4):
    width = int(frame.shape[1] * scalar)
    height = int(frame.shape[0] * scalar)
    dim = (width, height)

    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


def noiseReduction(image, iterationCount=6):
    mask = np.zeros(image.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    forground = np.zeros((1, 65), np.float64)
    rectangle = (5, 5, image.shape[1], image.shape[0])
    # rectangle = (5, 5, 100, 100)
    ret = cv.grabCut(image, mask, rectangle, background, forground, iterationCount, cv.GC_INIT_WITH_RECT)
    maskPrime = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    background = image

    image = image * maskPrime[:, :, np.newaxis]
    background -= image
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    image += background
    # plt.imshow(image), plt.colorbar(), plt.show()
    return image


# TODO: make a better way to not get tripped up by objects while scanning could make it a
#  percentage requirement
def findAndCut(image, mode="shirt"):
    allZ = True
    y = image.shape[0] - 1
    x = image.shape[1] - 1
    lastIJ = 0

    if (mode == "shirt"):
        for i in range(20, image.shape[0]):
            ticker = 0
            for j in range(20, int(image.shape[1])):
                if list(image[(y - i), (x - j)]) != [255, 255, 255]:
                    lastIJ = (y - i) - 10
                    ticker += 1
            # TODO tinker with this percentage value
            if lastIJ != 0 and ticker >= int(x / 3):
                break
        image = image[0:lastIJ, 0:image.shape[1]]
    else:
        for i in range(20, image.shape[0]):
            ticker = 0
            for j in range(20, int(image.shape[1])):
                if list(image[i, j]) != [255, 255, 255]:
                    lastIJ = i
                    ticker += 1

            if lastIJ != 0 and ticker >= int(x / 4):
                lastIJ += 15
                break
        image = image[lastIJ:image.shape[0], 0:image.shape[1]]
    return image


def stitch(one, two):
    prime = cv.resize(one, (two.shape[1], two.shape[0]), interpolation=cv2.INTER_LINEAR)
    prime = np.concatenate((prime, two), axis=0)
    plt.imshow(prime), plt.show()
    return prime

def makeImage(photo):
    return cv.imread(photo)

# TODO must pass in jpeg
def saveImage(name):
    cv2.imwrite(f"{name}")
    return cv.imread(name)
# cv.imshow('new', vis)
# plt.imshow(vis), plt.show()
# cv.imwrite('notstylish.jpg', vis)
