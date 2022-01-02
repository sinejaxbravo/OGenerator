import itertools
import os
import sys
import time

import matplotlib.pyplot as plt
import progressbar
import tensorflow as tf
import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph, NearestNeighbors
from tensorflow.keras.preprocessing import image

from tensorflow.python.keras.models import Model

from sklearn.preprocessing import StandardScaler

# from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import NeuralNet

dir_large = "/Users/stuar/Desktop/TrainingData/unsupervised"
dir_small = "/Users/stuar/Desktop/TrainingData/unsupervised_small"


# kmeans = KMeans(num_clusters=6, init=)


def get_photos(directory):
    photo_list = []
    for d in os.listdir(directory):
        photo_list.append(f"{directory}/{d}")
    return photo_list


def pca(data):
    1
    # pca = PCA(n_components=100, random_state=22)
    # pca.fit(data)
    # x = pca.transform(data)
    # kMeans(x)
    # print("d shape: ", data.shape)
    # print("transform shape: ", x.shape)


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


def K_Nearest(data, model):
    X = StandardScaler().fit_transform(data)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    # knn = KNeighborsClassifier(n_neighbors=50)
    nbrs.kneighbors_graph()
    A = kneighbors_graph(X, 2, mode='connectivity', include_self=True)

    for x in get_pair_pred(model):

        print(nbrs.kneighbors(x))
    plt.scatter(A), plt.show()


# min samples means how many assignments there needs to be before something becomes a cluster
def DB_SCAN(data):
    X = StandardScaler().fit_transform(data)
    dbscan = DBSCAN(eps=1.5, min_samples=20)
    dbscan.fit(X)
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    labels = dbscan.labels_
    unique_labels = set(dbscan.labels_)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=4,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=2,
        )

    # knn = KNeighborsClassifier(n_neighbors=50)
    # knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
    # plt.scatter(knn.kneighbors_graph(X))

    plt.show()
    # K_Nearest(dbscan)


def model():
    res = NeuralNet.oldCNN()
    # vgg = VGG16()
    model = res
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    photos = get_photos(dir_small)
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

    names = np.array(list(data.keys()))
    feats = np.array(list(data.values()))
    feats = feats.reshape((feats.shape[0], feats.shape[2]))
    print("Shape of our features for now: ", feats.shape)
    DB_SCAN(feats)
    K_Nearest(feats, model)


model()
