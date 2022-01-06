import itertools
import os
import sys
import time
from shutil import copyfile

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


fash = ["./clothes/pair/4.jpg", "./clothes/pair/5.jpg",
        "./clothes/pair/7.jpg", "./clothes/pair/8.jpg",
        "./clothes/pair/10.jpg","./clothes/pair/18.jpg","./clothes/pair/23.jpg",
        "./clothes/pair/56.jpg","./clothes/pair/68.jpg","./clothes/pair/69.jpg",
        "./clothes/pair/70.jpg","./clothes/pair/79.jpg","./clothes/pair/102.jpg"]

not_fash = ["./clothes/pair/2.jpg", "./clothes/pair/21.jpg",
        "./clothes/pair/22.jpg", "./clothes/pair/64.jpg",
        "./clothes/pair/49.jpg","./clothes/pair/50.jpg","./clothes/pair/51.jpg",
        "./clothes/pair/132.jpg","./clothes/pair/135.jpg","./clothes/pair/127.jpg",
        "./clothes/pair/178.jpg","./clothes/pair/181.jpg"]


def Affinity(data, labels, mode="x", title="Affinity", scalar=1.):
    cut = []
    X = StandardScaler().fit_transform(data)
    model = AffinityPropagation(damping=0.9, random_state=None)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    print("AFFINITY:")
    # print(clusters)
    if mode != "x":
        for i in range(X.shape[0]):
            if X[i, 0] < np.average(X[:, 0]) - np.std(X[:, 0]) * scalar and X[i, 1] > np.average(X[:, 1]) + np.std(X[:, 0]) * scalar:
                cut.append(labels[i])
            if labels[i] in fash:
                plt.scatter(X[i, 0], X[i, 1], c="red")
                # print("FASH SET--", labels[i])
                # print(X[i, 0], X[i, 1], "\n")
            elif labels[i] in not_fash and mode == "u":
                plt.scatter(X[i, 0], X[i, 1], c="purple")
                # print("NOT FASH SET--", labels[i])
                # print(X[i, 0], X[i, 1], "\n")

        print(f"Scalar: {scalar}")
        cut.sort()
        cut.sort(key=len)
        output_path = output[scalar]
        print(output_path)
        for i in cut:
            t = i[-10:len(i)]
            print(output_path+t[t.index("/"):t.index(".")]+".jpg")
            copyfile(i, output_path+t[t.index("/"):t.index(".")]+".jpg")

    i = 0
    if mode == "x":
        for cluster in clusters:
            print(labels[i])
            print(X[i], "\n\n")
            i += 1
            # get row indexes for samples with this cluster
            row_ix = np.where(yhat == cluster)
            # create scatter of these samples
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            print(X[row_ix, 0], X[row_ix, 1], "\n")
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
def DB_SCAN(data, labels, mode="x", title="DB SCAN"):
    X = StandardScaler().fit_transform(data)
    print(X.shape[1], X.shape[0])
    dbscan = DBSCAN(eps=1, min_samples=20)
    model = dbscan.fit(X)
    yhat = dbscan.fit_predict(X)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    print("DB_SCAN:")
    print(clusters)
    # create scatter plot for samples from each cluster
    if mode != "x":
        for i in range(X.shape[0]):
            if labels[i] in fash:
                plt.scatter(X[i, 0], X[i, 1], c="red")
                print("FASH SET--", labels[i])
                print(X[i, 0], X[i, 1], "\n")
            elif labels[i] in not_fash and mode == "u":
                plt.scatter(X[i, 0], X[i, 1], c="purple")
                print("NOT FASH SET--", labels[i])
                print(X[i, 0], X[i, 1], "\n")

    i = 0
    if mode == "x":
        for cluster in clusters:
            print(labels[i])
            print(X[i], "\n\n")
            i += 1
            # get row indexes for samples with this cluster
            row_ix = np.where(yhat == cluster)
            # create scatter of these samples
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
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


def get_pair_pred(model):
    features = []

    for filename in os.listdir("clothes/pair"):
        imag = image.load_img(f"clothes/pair/{filename}", target_size=(224, 224))
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
    res = NeuralNet.fashion_CNN()
    # vgg = VGG16()
    model = res
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    photos = get_photos(dir_small)
    pairs = get_pred_photos(dir_pred)
    pairs = pairs
    pairs, pair_feats = features_lists(pairs, model)
    # DB_SCAN(pair_feats, pair)
    # DB_SCAN(pair_feats, pair, "m")
    # DB_SCAN(pair_feats, pair, "u")
    # Mean_Shift(pair_feats, pair)
    # Affinity(pair_feats, pair)
    Affinity(pair_feats, pairs, "u", ".5 ", .5)
    Affinity(pair_feats, pairs, "m")
    Affinity(pair_feats, pairs, "u", ".5 ", 1.5)
    Affinity(pair_feats, pairs, "u", ".5 ", 0)



    # pairsN, pair_featsN = good_bad_outfits(model, not_fash)
    # pair, pair_feats = good_bad_outfits(model, fash)
    # DB_SCAN(pair_feats, pair, "Liked DB Scan")
    # DB_SCAN(pair_featsN, pairsN, "Not Liked DB Scan")
    # # Mean_Shift(pair_feats, pair)
    # Affinity(pair_feats, pair, "Liked Affinity")
    #
    #
    # # Mean_Shift(pair_feats, pair)
    # Affinity(pair_featsN, pairsN, "Not Liked Affinity")
    # Optics(pair_feats, pair)
    # Optics(pair_featsN, pairsN)
    # K_Nearest(pair_feats, model)
    # i = 0
    # data = {}
    # with progressbar.ProgressBar(max_value=len(photos)) as bar:
    #     for pic in photos:
    #         bar.update(i)
    #         time.sleep(.02)
    #         imag = image.load_img(pic, target_size=(224, 224))
    #         img_array = image.img_to_array(imag)
    #         img_batch = np.expand_dims(img_array, 0)
    #         extract = extraction(model, img_batch)
    #         data[pic] = extract
    #         i += 1

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
