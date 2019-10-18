import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def generate_cluster():
    data = pd.DataFrame(np.zeros((5000, 3)), columns=['x1', 'x2'])

    # Let's make up some noisy XOR data to use to build our binary classifier
    for i in range(len(data.index)):
        x1 = 1.0 * random.randint(0,1)
        x2 = 1.0 * random.randint(0,1)
        x1 = x1 + 0.15 * np.random.normal()
        x2 = x2 + 0.15 * np.random.normal()
        data.iloc[i,0] = x1
        data.iloc[i,1] = x2

    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]

    X = np.matrix(X.values)

