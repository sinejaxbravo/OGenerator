import itertools
import os
import sys
import time
from collections import Set

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

dir_pred = "./clothes/pairs"

fash = ["./clothes/pairs/4.jpg", "./clothes/pairs/5.jpg",
        "./clothes/pairs/7.jpg", "./clothes/pairs/8.jpg",
        "./clothes/pairs/10.jpg","./clothes/pairs/18.jpg","./clothes/pairs/23.jpg",
        "./clothes/pairs/56.jpg","./clothes/pairs/68.jpg","./clothes/pairs/69.jpg",
        "./clothes/pairs/70.jpg","./clothes/pairs/79.jpg","./clothes/pairs/102.jpg"]

not_fash = ["./clothes/pairs/2.jpg", "./clothes/pairs/21.jpg",
        "./clothes/pairs/22.jpg", "./clothes/pairs/64.jpg",
        "./clothes/pairs/49.jpg","./clothes/pairs/50.jpg","./clothes/pairs/51.jpg",
        "./clothes/pairs/132.jpg","./clothes/pairs/135.jpg","./clothes/pairs/127.jpg",
        "./clothes/pairs/178.jpg","./clothes/pairs/181.jpg"]


def Affinity(data, labels, title="Affinity"):

    X = StandardScaler().fit_transform(data)
    model = AffinityPropagation(damping=0.9, random_state=None)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    print("AFFINITY:")
    print(clusters)

    for i in range(X.shape[0]):
        if labels[i] in fash:
            print("FASH SET--", labels[i])
            print(X[i, 0], X[i, 1], "\n")
        elif labels[i] in not_fash:
            print("NOT FASH SET--", labels[i])
            print(X[i, 0], X[i, 1], "\n")

    i = 0
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        print(labels[i])
        print(X[i], "\n\n")

        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        if labels[i] in fash:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], c="red")
        elif labels[i] in not_fash:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], c="black")
        else:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], c="blue")
        print(X[row_ix, 0], X[row_ix, 1], "\n")
        i += 1
    # show the plot
    plt.title(f"{title}")
    plt.show()


def Optics(data, labels):
    X = StandardScaler().fit_transform(data)
    model = OPTICS(eps=0.8, min_samples=10)
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    print("OPTICS:")
    # create scatter plot for samples from each cluster
    i = 0
    for cluster in clusters:
        print(labels[i])
        print(X[i], "\n\n")
        i += 1
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    plt.title("Optics")
    plt.show()


def Mean_Shift(data, labels):
    X = StandardScaler().fit_transform(data)
    model = MeanShift()
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    i = 0
    print("MEAN_SHIFT:")
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        print(labels[i])
        print(X[i], "\n\n")
        i += 1
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    plt.title("Mean Shift")
    plt.show()


def K_Means(data, labels):
    X = StandardScaler().fit_transform(data)
    model = KMeans(n_clusters=2)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    i = 0
    print("K_MEANS:")
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        print(labels[i])
        print(X[i], "\n\n")
        i += 1
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    plt.title("K Means")
    plt.show()


def pca(data):
    1
    # pca = PCA(n_components=100, random_state=22)
    # pca.fit(data)
    # x = pca.transform(data)
    # kMeans(x)
    # print("d shape: ", data.shape)
    # print("transform shape: ", x.shape)


def K_Nearest(data, model):
    X = StandardScaler().fit_transform(data)
    nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(X)
    # knn = KNeighborsClassifier(n_neighbors=50)
    A = kneighbors_graph(X, 8, mode='connectivity', include_self=True)

    # for x in get_pair_pred(model):
    #     print(nbrs.kneighbors(x))
    # plt.scatter(A), plt.show()


# min samples means how many assignments there needs to be before something becomes a cluster
def DB_SCAN(data, labels, title="DB SCAN"):
    X = StandardScaler().fit_transform(data)
    dbscan = DBSCAN(eps=.5, min_samples=3)
    model = dbscan.fit(X)
    yhat = dbscan.fit_predict(X)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    print("DB_SCAN:")
    print(clusters)
    # create scatter plot for samples from each cluster
    for i in range(X.shape[0]):
        if labels[i] in fash:
            print("FASH SET--", labels[i])
            print(X[i, 0], X[i, 1], "\n")
        elif labels[i] in not_fash:
            print("NOT FASH SET--", labels[i])
            print(X[i, 0], X[i, 1], "\n")

    i = 0
    for cluster in clusters:
        print(labels[i])
        print(X[i], "\n\n")
        i += 1
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        if labels[i] in fash:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], c="red")
        elif labels[i] in not_fash:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], c="black")
        else:
            plt.scatter(X[row_ix, 0], X[row_ix, 1], c="blue")
        print(X[row_ix, 0], X[row_ix, 1], "\n")
    # show the plot

    # knn = KNeighborsClassifier(n_neighbors=50)
    # knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
    # plt.scatter(knn.kneighbors_graph(X))
    plt.title(f"{title}")
    plt.show()
    # K_Nearest(dbscan)


def get_photos(directory):
    photo_list = []
    for d in os.listdir(directory):
        photo_list.append(f"{directory}/{d}")
    return photo_list


def get_pred_photos(directory):
    photo_list = []
    for d in os.listdir(directory):
        photo_list.append(f"{directory}/{d}")
    return photo_list


def pred_features_lists(directory, model):
    data = {}
    for pic in directory:

        imag = image.load_img(pic, target_size=(224, 224))
        img_array = image.img_to_array(imag)
        img_batch = np.expand_dims(img_array, 0)
        extract = extraction(model, img_batch)
        data[pic] = extract

    names = np.array(list(data.keys()))
    feats = np.array(list(data.values()))
    feats = feats.reshape((feats.shape[0], feats.shape[2]))
    return names, feats


def extraction(model, image):
    # print(img.shape)
    feature = model.predict(image, use_multiprocessing=True)
    return feature


def get_pair_pred(model):
    features = []
    for filename in os.listdir("clothes/pairs"):
        imag = image.load_img(f"clothes/pairs/{filename}", target_size=(224, 224))
        img_array = image.img_to_array(imag)
        img_batch = np.expand_dims(img_array, 0)
        prediction = model.predict(img_batch)
        features.append(prediction)
    return features


def good_bad_outfits(model, set):
    features = []
    data = {}
    for filename in set:
        imag = image.load_img(f"{filename}", target_size=(224, 224))
        img_array = image.img_to_array(imag)
        img_batch = np.expand_dims(img_array, 0)
        prediction = model.predict(img_batch)
        features.append(prediction)
    for pic in set:
        imag = image.load_img(pic, target_size=(224, 224))
        img_array = image.img_to_array(imag)
        img_batch = np.expand_dims(img_array, 0)
        extract = extraction(model, img_batch)
        data[pic] = extract

    names = np.array(list(data.keys()))
    feats = np.array(list(data.values()))
    feats = feats.reshape((feats.shape[0], feats.shape[2]))
    return names, feats



def model():
    res = NeuralNet.oldCNN()
    # vgg = VGG16()
    model = res
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    photos = get_photos(dir_small)
    pairs = get_pred_photos(dir_pred)
    pairs, pair_feats = pred_features_lists(pairs, model)
    DB_SCAN(pair_feats, pairs)
    # Mean_Shift(pair_feats, pairs)
    Affinity(pair_feats, pairs)

    pairsN, pair_featsN = good_bad_outfits(model, not_fash)
    pairs, pair_feats = good_bad_outfits(model, fash)
    DB_SCAN(pair_feats, pairs, "Liked DB Scan")
    DB_SCAN(pair_featsN, pairsN, "Not Liked DB Scan")
    # Mean_Shift(pair_feats, pairs)
    Affinity(pair_feats, pairs, "Liked Affinity")


    # Mean_Shift(pair_feats, pairs)
    Affinity(pair_featsN, pairsN, "Not Liked Affinity")
    Optics(pair_feats, pairs)
    Optics(pair_featsN, pairsN)
    # K_Nearest(pair_feats, model)
    i = 0
    data = {}
    with progressbar.ProgressBar(max_value=len(photos)) as bar:
        for pic in photos:
            bar.update(i)
            time.sleep(.02)
            imag = image.load_img(pic, target_size=(224, 224))
            img_array = image.img_to_array(imag)
            img_batch = np.expand_dims(img_array, 0)
            extract = extraction(model, img_batch)
            data[pic] = extract
            i += 1

    # names = np.array(list(data.keys()))
    # feats = np.array(list(data.values()))
    # feats = feats.reshape((feats.shape[0], feats.shape[2]))
    # print("Shape of our features for now: ", feats.shape)
    # DB_SCAN(feats, names)
    # Mean_Shift(feats, names)
    # Affinity(feats, names)
    # Optics(feats, names)
    # K_Nearest(feats, model)


model()
