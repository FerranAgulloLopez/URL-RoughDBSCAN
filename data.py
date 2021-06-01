from sklearn.datasets import make_moons, make_blobs


def generate_blobs_dataset(samples, centers, random_seed):
    X, _ = make_blobs(n_samples=samples, centers=centers, cluster_std=0.60, random_state=random_seed)
    return X


def generate_moons_dataset(samples, random_seed):
    Xmoon, _ = make_moons(samples, noise=.05, random_state=random_seed)
    return Xmoon
