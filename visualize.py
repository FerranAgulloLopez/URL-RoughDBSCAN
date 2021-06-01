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

def plot_multiple_clustering_2d(nrows, ncols, X, ys, title, ylabels, xlabels, output):
    index = 0
    if nrows > 1:
        fig, axs = plt.subplots(nrows, ncols, figsize=(nrows * 10, ncols * 10))
        for row in range(nrows):
            for col in range(ncols):
                y = ys[index]
                unique = np.unique(y, return_counts=False)
                for label_index in range(len(unique)):
                    label = unique[label_index]
                    X_index = X[y == label]
                    axs[row, col].scatter(X_index[:, 0], X_index[:, 1], label='cluster_' + str(label) if label != -1 else 'noise')
                index += 1
        for index, label in enumerate(ylabels):
            axs[index, 0].set_ylabel(label, fontsize=6 * 6)
        for index, label in enumerate(xlabels):
            axs[0, index].set_title(label, fontsize=6 * 6)
    else:
        fig, axs = plt.subplots(nrows, ncols, figsize=(nrows * 16, ncols * 1))
        for col in range(ncols):
            y = ys[index]
            unique = np.unique(y, return_counts=False)
            for label_index in range(len(unique)):
                label = unique[label_index]
                X_index = X[y == label]
                axs[col].scatter(X_index[:, 0], X_index[:, 1], label='cluster_' + str(label) if label != -1 else 'noise')
            index += 1
        for index, label in enumerate(xlabels):
            axs[index].set_title(label, fontsize=6 * 2)
    plt.savefig(output, bbox_inches='tight')
    plt.show()


def compare_multiple_lines(visualize, x_array, lines, path, legend=True, ylabel='Loss', title=None, ylim=None):
    # pre: the len of the input arrays must be equal to the number of epochs
    #      the input arrays must have two dimensions (data, label)
    fig, ax = plt.subplots()
    for line in lines:
        label, data = line
        ax.plot(x_array[:len(data)], data, label=label)
    if legend:
        ax.legend(loc='best', shadow=True)
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel("Number samples")
    plt.ylabel(ylabel)
    plt.savefig(path)
    if visualize:
        plt.show()
