import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

import Directories


def rescaleImage(frame, scalar=0.4):
    width = int(frame.shape[1] * scalar)
    height = int(frame.shape[0] * scalar)
    dim = (width, height)

    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


def getColor(image):
    init = image[20, int(image.shape[1] / 2)]
    print(init)
    plt.imshow(image), plt.show()
    for x in range(image.shape[0]):
        res = image[x, int(image.shape[1] / 2)] - init
        if np.average(res) > 100:
            print(res)
            return image[x, int(image.shape[1] / 2)]


def makeMask(image, color):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if np.average(image[x, y] - color) > 50:
                image[x, y] = [255, 255, 255]


def noiseReduction(image, color, iterationCount=6):
    mask = np.zeros(image.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    forground = np.zeros((1, 65), np.float64)
    rectangle = (6, 6, image.shape[1], image.shape[0])

    # rectangle = (5, 5, 100, 100)
    ret = cv.grabCut(image, mask, rectangle, background, forground, iterationCount, cv.GC_INIT_WITH_RECT)
    maskPrime = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    background = image

    image = image * maskPrime[:, :, np.newaxis]
    background -= image
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    image += background
    plt.imshow(image), plt.show()
    return image


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


def makeImage(photo):
    return cv.imread(photo)

def saveImage(name):
    cv2.imwrite(f"{name}")
    return cv.imread(name)

def make_square():
    path = Directories.path_dual
    for folder in os.listdir(path):
        if folder[0] != "s":
            for filename in os.listdir(path + f"/{folder}"):
                p = path + "\\" + folder + "\\" + filename
                print(p)
                photo = cv2.imread(p)
                print(path + f"\\s{folder}" + filename)
                photo = cv2.resize(photo, (1600, 1600), interpolation=cv2.INTER_AREA)
                plt.imshow(photo), plt.show()
                square = photo[600:1000, 520:1080]
                square = cv2.resize(square, (240, 120), interpolation=cv2.INTER_LINEAR)
                pants = photo[1500:photo.shape[0], 520:700]
                pants = np.concatenate((photo[1500:photo.shape[0], 520:700], pants), axis=0)
                pants = cv2.resize(pants, (square.shape[1], square.shape[0]), interpolation=cv2.INTER_AREA)
                print(pants.shape, square.shape)
                prime = np.concatenate((square, pants), axis=0)
                prime2 = np.concatenate((pants, square), axis=0)
                plt.imshow(prime), plt.show()
                plt.imshow(prime2), plt.show()
                cv2.imwrite(path + f"\\s{folder}\\" + filename, prime2)
                cv2.imwrite(path + f"\\s{folder}\\" + "flip_" + filename, prime)