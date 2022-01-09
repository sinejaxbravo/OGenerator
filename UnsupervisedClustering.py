import itertools
import os
import sys
import time
from shutil import copyfile
import pickle

import matplotlib.pyplot as plt
import progressbar
import tensorflow as tf
import numpy as np

from sklearn.cluster import KMeans, DBSCAN, MeanShift, OPTICS, AffinityPropagation
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph, NearestNeighbors
from tensorflow.keras.preprocessing import image

from tensorflow.python.keras.models import Model

from sklearn.preprocessing import StandardScaler

# from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import NeuralNet

dir_large = "/Users/stuar/Desktop/TrainingData/unsupervised"
dir_small = "/Users/stuar/Desktop/TrainingData/unsupervised_small"

dir_pred = "clothes/pair"

dir_1 = "./clothes/predicted_1"
dir_15 = "./clothes/predicted_1dot5"
dir_5 = "./clothes/predicted_dot5"
dir_zip = "./clothes/predicted_zip"

output = {1: dir_1, 1.5: dir_15, .5: dir_5, 0: dir_zip}

# for o in output:
#     for x in output[o]:
#         os.remove(x)


# fash = ["./clothes/pair/4.jpg", "./clothes/pair/5.jpg",
#         "./clothes/pair/7.jpg", "./clothes/pair/8.jpg",
#         "./clothes/pair/10.jpg", "./clothes/pair/18.jpg", "./clothes/pair/23.jpg",
#         "./clothes/pair/56.jpg", "./clothes/pair/68.jpg", "./clothes/pair/69.jpg",
#         "./clothes/pair/70.jpg", "./clothes/pair/79.jpg", "./clothes/pair/102.jpg"]
#
# not_fash = ["./clothes/pair/2.jpg", "./clothes/pair/21.jpg",
#             "./clothes/pair/22.jpg", "./clothes/pair/64.jpg",
#             "./clothes/pair/49.jpg", "./clothes/pair/50.jpg", "./clothes/pair/51.jpg",
#             "./clothes/pair/132.jpg", "./clothes/pair/135.jpg", "./clothes/pair/127.jpg",
#             "./clothes/pair/178.jpg", "./clothes/pair/181.jpg"]


def Affinity(data, labels, mode="x", title="Affinity", scalar=1.):
    cut = []
    X = StandardScaler().fit_transform(data)
    print(X)
    model = AffinityPropagation(damping=0.9, random_state=None)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    print("AFFINITY:")
    # print(clusters)
    to_ret = []
    if mode != "x":
        for i in range(X.shape[0]):
            to_ret.append([X[i, 0], X[i, 1]])
            if X[i, 0] < np.average(X[:, 0]) - np.std(X[:, 0]) * scalar and X[i, 1] > np.average(X[:, 1]) + np.std(
                    X[:, 1]) * scalar:
                cut.append(labels[i])

            # plt.scatter(X[i, 0], X[i, 1], c="red")
            # print("FASH SET--", labels[i])
            # print(X[i, 0], X[i, 1], "\n")
        # plt.show()
        return to_ret

    if mode == "x":
        stats = (np.average(X[:, 0]), np.std(X[:, 0]), np.average(X[:, 1]), np.std(X[:, 0]))
        return stats


def PCA_And_K_Means(data, ):
    pca = PCA(n_components=100, random_state=22)
    pca.fit(data)
    x = pca.transform(data)
    kmeans = KMeans(n_clusters=3, n_jobs=-1, random_state=22)
    label = kmeans.fit(x)
    u_labels = np.unique(label)
    pickle.dump(kmeans, open("kmeans_pairs.pkl", "wb"))

    # plotting the results
    for i in u_labels:
        plt.scatter(x[label == i, 0], x[label == i, 1], label=i)
    plt.legend()
    plt.show()



def get_photos(directory):
    photo_list = []
    for d in os.listdir(directory):
        photo_list.append(f"{directory}/{d}")
    return photo_list


def features_lists(directory, model):
    data = {}
    i = 0
    with progressbar.ProgressBar(max_value=len(directory)) as bar:
        for pic in directory:
            bar.update(i)
            time.sleep(.02)
            imag = image.load_img(pic, target_size=(224, 224))
            img_array = image.img_to_array(imag)
            img_batch = np.expand_dims(img_array, 0)
            extract = extraction(model, img_batch)
            data[pic] = extract
            i += 1
        names = np.array(list(data.keys()))
        feats = np.array(list(data.values()))
        feats = feats.reshape((feats.shape[0], feats.shape[2]))
        return names, feats


def extraction(model, image):
    # print(img.shape)
    feature = model.predict(image, use_multiprocessing=True)
    return feature


def get_photos_and_features(model):
    features = []
    data = {}
    for filename in os.listdir("clothes/pair"):
        imag = image.load_img(f"clothes/pair/{filename}", target_size=(224, 224))
        img_array = image.img_to_array(imag)
        img_batch = np.expand_dims(img_array, 0)
        prediction = model.predict(img_batch)
        extract = extraction(model, img_batch)
        data[filename] = extract
        features.append(prediction)

    names = np.array(list(data.keys()))
    feats = np.array(list(data.values()))
    feats = feats.reshape((feats.shape[0], feats.shape[2]))
    return names, feats


def train():
    res = NeuralNet.fashion_CNN()
    res = Model(inputs=res.inputs, outputs=res.layers[-2].output)
    return res


def model(images, res, mode="m", pca=False):
    # vgg = VGG16()
    model = res
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    if pca:
        pair, pair_feats = get_photos_and_features(res)
        PCA_And_K_Means(pair_feats)

    else:
        # if mode == "m":
        #     images = get_photos(images)
    # pairs = get_pred_photos(dir_pred, mode, images)
        pairs, pair_feats = features_lists(images, model)

        return Affinity(pair_feats, pairs, mode)
    # Affinity(pair_feats, pairs, "u", ".5 ", 1.5)
    # Affinity(pair_feats, pairs, "u", ".5 ", 0)
