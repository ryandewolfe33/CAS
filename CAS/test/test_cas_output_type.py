import pytest
import numpy as np
import scipy.sparse as sp
from CAS import CASPostProcesser


def test_cas_array_only_remove(sbm_leiden):
    adjacency, predict = sbm_leiden[0], sbm_leiden[1]
    cpp = CASPostProcesser(only_remove=True)
    out = cpp.fit_predict(predict, adjacency)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert len(out) == len(predict)
    assert np.min(out) >= -1
    assert np.max(out) <= np.max(predict)


def test_cas_array_only_remove_and_sparse(sbm_leiden):
    adjacency, predict = sbm_leiden[0], sbm_leiden[1]
    cpp = CASPostProcesser(only_remove=True, sparse_output=True)
    out = cpp.fit_predict(predict, adjacency)
    assert isinstance(out, sp.csr_matrix)
    assert out.shape == (np.max(predict) + 1, len(predict))
    assert all(out.data)


def test_cas_array_add_and_remove(sbm_leiden):
    adjacency, predict = sbm_leiden[0], sbm_leiden[1]
    cpp = CASPostProcesser(only_remove=False)
    out = cpp.fit_predict(predict, adjacency)
    assert isinstance(out, sp.csr_matrix)
    assert out.shape == (np.max(predict) + 1, len(predict))
    assert all(out.data)


def test_cas_sparse_only_remove(sbm_sparse):
    adjacency, predict = sbm_sparse[0], sbm_sparse[1]
    cpp = CASPostProcesser(only_remove=True)
    out = cpp.fit_predict(predict, adjacency)
    assert isinstance(out, sp.csr_matrix)
    assert out.shape == predict.shape
    assert all(out.data)


def test_cas_sparse_add_and_remove(sbm_sparse):
    adjacency, predict = sbm_sparse[0], sbm_sparse[1]
    cpp = CASPostProcesser(only_remove=False)
    out = cpp.fit_predict(predict, adjacency)
    assert isinstance(out, sp.csr_matrix)
    assert out.shape == predict.shape
    assert all(out.data)
