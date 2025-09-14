# ===========================
#  Testing (session) Fixture
# ==========================

import pytest
import numpy as np
import scipy.sparse as sp
import sknetwork as sn
from CAS import CASPostProcesser

# Global RNG
SEED = 5
np.random.seed(SEED)

@pytest.fixture(scope="session")
def sbm_leiden():
    adjacency = sn.data.block_model([25]*8)
    predict = sn.clustering.Leiden().fit_predict(adjacency)
    return adjacency, predict
