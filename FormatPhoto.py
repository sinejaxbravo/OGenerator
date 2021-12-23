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


def noiseReduction(image, imageName, iterationCount=0):
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
    plt.imshow(image), plt.colorbar(), plt.show()
    # cv2.imwrite(f'{imageName}', image)


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
                    lastIJ = (y - i) - 5
                    ticker += 1
            # TODO tinker with this percentage value
            if lastIJ != 0 and ticker >= int(x / 3):
                print(lastIJ)
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


shirts = ['IMG_0655.jpg', 'IMG_0663.jpg', 'IMG_0664.jpg']
toFormat = []
for shirt in toFormat:
    shirt_image = cv.imread(shirt)
    shirt_image = rescaleImage(shirt_image)
    # cv.imshow('Fashion', shirt_image)
    noiseReduction(shirt_image, shirt)

one = rescaleImage(cv.imread(shirts[0]))
two = rescaleImage(cv.imread(shirts[2]))
pants = rescaleImage(cv.imread(shirts[1]))


one = findAndCut(two, "shirt")
plt.imshow(one), plt.show()
pants = findAndCut(pants, "")

prime = cv.resize(one, (pants.shape[1], pants.shape[0]), interpolation=cv2.INTER_LINEAR)

vis = np.concatenate((prime, pants), axis=0)
cv.imshow('new', vis)
plt.imshow(vis), plt.show()
cv.imwrite('notstylish.jpg', vis)

