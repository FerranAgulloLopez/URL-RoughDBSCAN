from sklearn.datasets import make_moons, make_blobs


def generate_blobs_dataset(samples, centers):
    X, _ = make_blobs(n_samples=samples, centers=centers, cluster_std=0.60)
    return X


def generate_moons_dataset(samples):
    X, _ = make_moons(samples, noise=.05)
    return X
