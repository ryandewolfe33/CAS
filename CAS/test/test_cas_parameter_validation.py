import pytest
from CAS import CASPostProcesser


def test_cas_score(sbm_leiden):
    adjacency, predict = sbm_leiden[0], sbm_leiden[1]
    # Check implemented scores work
    cpp = CASPostProcesser(score="ief")
    cpp.fit(predict, adjacency)
    cpp = CASPostProcesser(score="nief")
    cpp.fit(predict, adjacency)
    cpp = CASPostProcesser(score="p")
    cpp.fit(predict, adjacency)
    # Check that other scores raise a ValueError
    cpp = CASPostProcesser(score="other")
    with pytest.raises(ValueError):
        cpp.fit(predict, adjacency)


def test_max_per_round(sbm_leiden):
    adjacency, predict = sbm_leiden[0], sbm_leiden[1]
    cpp = CASPostProcesser(max_per_round=-1)
    with pytest.raises(ValueError):
        cpp.fit(predict, adjacency)
    cpp = CASPostProcesser(max_per_round=3.5)
    with pytest.raises(ValueError):
        cpp.fit(predict, adjacency)


def test_max_rounds(sbm_leiden):
    adjacency, predict = sbm_leiden[0], sbm_leiden[1]
    cpp = CASPostProcesser(max_rounds=-1)
    with pytest.raises(ValueError):
        cpp.fit(predict, adjacency)
    cpp = CASPostProcesser(max_rounds=3.5)
    with pytest.raises(ValueError):
        cpp.fit(predict, adjacency)


def test_only_remove(sbm_leiden):
    adjacency, predict = sbm_leiden[0], sbm_leiden[1]
    cpp = CASPostProcesser(only_remove=3.5)
    with pytest.raises(ValueError):
        cpp.fit(predict, adjacency)


def test_sparse_output(sbm_leiden):
    adjacency, predict = sbm_leiden[0], sbm_leiden[1]
    cpp = CASPostProcesser(sparse_output=3.5)
    with pytest.raises(ValueError):
        cpp.fit(predict, adjacency)
