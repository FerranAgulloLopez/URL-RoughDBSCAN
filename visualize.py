import matplotlib.pyplot as plt
import numpy as np


def plot_clustering_2d(X, y):
    fig, ax = plt.subplots()
    unique = np.unique(y, return_counts=False)
    for index in range(len(unique)):
        label = unique[index]
        X_index = X[y == label]
        plt.scatter(X_index[:, 0], X_index[:, 1], label='cluster_' + str(label) if label != -1 else 'noise')
    ax.set_xlabel('1rst dim. coords')
    ax.set_ylabel('2nd dim. coords')
    ax.set_title('Dataset clustering split')
    plt.legend()
    plt.show()
