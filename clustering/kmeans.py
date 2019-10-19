'''
Source:
https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/

'''

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy



def generate_cluster():
    data = pd.DataFrame(np.zeros((5000, 2)), columns=['x1', 'x2'])

    # Let's make up some noisy XOR data to use to build our binary classifier
    for i in range(len(data.index)):
        x1 = 1.0 * random.randint(0,1)
        x2 = 1.0 * random.randint(0,1)
        x1 = x1 + 0.15 * np.random.normal()
        x2 = x2 + 0.15 * np.random.normal()
        data.iloc[i,0] = x1
        data.iloc[i,1] = x2

    cols = data.shape[1]
    X = data.iloc[:,0:cols].values
    return X

def dist(a, b, ax = 1):
    return np.linalg.norm(a - b, axis=ax)

class K_means:
    def __init__(self, k=4, tol = 0.001, max_iter=300):
        self.k = k 
        self.tol = tol 
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classification = {}

            for i in range(self.k):
                self.classification[i] = []
            
            for sample in data:
                distances = [dist(sample, self.centroids[i]) for i in range(self.k)]
                c = np.argmin(distances)
                self.classifications[c].append(sample)
            
            prev_centroid = dict(self.centroid)

            for i in self.classification:
                self.centroid[i] = np.avarage(self.classification[i], axis=0)

            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroid[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = True
            
            if optimized:
                break

    def predict(self, data):
        distances = [dist(sample, self.centroids[i]) for i in range(k)]
        c = np.argmin(distances)
        return c



if __name__ == "__main__":
    X = generate_cluster()   
    colors = 10*["g","r","c","b","k"]
    clf = K_means()
    clf.fit(X)

    for centroid in clf.centroids:
        plt.scatter(clf.centroid[centroid][0], clf.centroid[centroid][1],
                    marker='o', color='k', s=150, linewidth=5)
    
    for c in clf.classification:
        color = colors[c]
        for sample in clf.classification[c]:
            plt.scatter(sample[0], sample[1], marker="x", 
            color=color, s=150, linewidths=5)
