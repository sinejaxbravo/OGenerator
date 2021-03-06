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
from tensorflow.keras.preprocessing import image

from tensorflow.python.keras.models import Model

from sklearn.preprocessing import StandardScaler

# from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import NeuralNet


# dir_large is for training on our larger data set
dir_large = "/Users/stuar/Desktop/TrainingData/unsupervised"
dir_small = "/Users/stuar/Desktop/TrainingData/unsupervised_small"

# dir_pred is our prediction folder for pairs of items
dir_pred = "clothes/pair"

# dir_standardDeviation so dir_1 is 1 S.D. away. Used for clsutering
dir_1 = "./clothes/predicted_1"
dir_15 = "./clothes/predicted_1dot5"
dir_5 = "./clothes/predicted_dot5"
dir_zip = "./clothes/predicted_zip"


output = {1: dir_1, 1.5: dir_15, .5: dir_5, 0: dir_zip}

""""
Affinity:
    method that takes in:
        data which are the extracted features
        labels which are used to reference the class the features belong to
        
        

"""

def Affinity(data, labels, mode="x", title="Affinity", scalar=1.):
    cut = []
    X = StandardScaler().fit_transform(data)
    print(X)
    model = AffinityPropagation(damping=0.9, random_state=None)
    model.fit(X)
    yhat = model.predict(X)
    clusters = np.unique(yhat)
    print("AFFINITY:")
    to_ret = []
    if mode != "x":
        for i in range(X.shape[0]):
            to_ret.append([X[i, 0], X[i, 1]])
            if X[i, 0] < np.average(X[:, 0]) - np.std(X[:, 0]) * scalar and X[i, 1] > np.average(X[:, 1]) + np.std(
                    X[:, 1]) * scalar:
                cut.append(labels[i])
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


def model(images, res, mode="default", pca=False):
    # vgg = VGG16()
    model = res
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    if pca:
        pair, pair_feats = get_photos_and_features(res)
        PCA_And_K_Means(pair_feats)

    else:
        pairs, pair_feats = features_lists(images, model)

        return Affinity(pair_feats, pairs, mode)

