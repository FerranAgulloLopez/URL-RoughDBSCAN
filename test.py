from src.algorithms.types.CountedLeaders import CountedLeaders
from src.algorithms.types.RoughDBSCAN import RoughDBSCAN
import numpy as np


X = np.asarray([[0], [4], [1], [4], [50], [5]])
algorithm = RoughDBSCAN(2, 4, 3)
algorithm.fit(X)