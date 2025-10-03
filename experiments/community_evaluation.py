import numpy as np
import scipy.sparse as sp
import sknetwork as sn
from numba import njit
import CAS


@njit
def _compute_conductance(
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
    labels_indptr,
    labels_indices,
):
    conductance = np.empty(len(labels_indptr) - 1)
    for label in range(len(labels_indptr) - 1):
        if labels_indptr[label] == labels_indptr[label + 1]:  # Empty cluster
            conductance[label] = np.nan
            continue
        nodes = labels_indices[labels_indptr[label] : labels_indptr[label + 1]]
        boundary_degree = 0
        total_degree = 0
        for node in nodes:
            neighbor_indices = np.arange(
                adjacency_indptr[node], adjacency_indptr[node + 1]
            )
            total_degree += np.sum(adjacency_data[neighbor_indices])
            # Add in label edge weights
            for index in neighbor_indices:
                neighbor = adjacency_indices[index]
                if neighbor not in nodes:
                    boundary_degree += adjacency_data[index]
        conductance[label] = boundary_degree / total_degree
    return conductance


def conductance(adjacency, labels):
    """Compute the conductance of each label.
    Conductance is the fraction of edge volumne exiting the cluster.
    Labels should be a csr matrix (labels x nodes).
    Returns a 1d numpy array of length labels.
    Empty clusters are given a score of 0."""
    return _compute_conductance(
        adjacency.indptr,
        adjacency.indices,
        adjacency.data,
        labels.indptr,
        labels.indices,
    )


def density(adjacency, labels):
    """Compute the internal density of each label.
    Density is the fraction of existing edges to possible edges.
    Labels should be a csr matrix (labels x nodes).
    Returns a 1d numpy array of length labels.
    Empty clusters are given a score of 0."""
    density = np.empty(labels.shape[0])
    for i in range(labels.shape[0]):
        nodes = labels.indices[labels.indptr[i] : labels.indptr[i + 1]]
        if len(nodes) < 2:
            density[i] = np.nan
            continue
        label_subgraph = adjacency[nodes][:, nodes]
        internal_edges = np.sum(label_subgraph.data) // 2
        density[i] = internal_edges / (len(nodes) * (len(nodes) - 1))
    return density


@njit
def _compute_expansion(
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
    labels_indptr,
    labels_indices,
):
    expansion = np.empty(len(labels_indptr) - 1)
    for label in range(len(labels_indptr) - 1):
        if labels_indptr[label] == labels_indptr[label + 1]:  # Empty cluster
            expansion[label] = np.nan
            continue
        nodes = labels_indices[labels_indptr[label] : labels_indptr[label + 1]]
        boundary_degree = 0
        for node in nodes:
            neighbor_indices = np.arange(
                adjacency_indptr[node], adjacency_indptr[node + 1]
            )
            # Add in label edge weights
            for index in neighbor_indices:
                neighbor = adjacency_indices[index]
                if neighbor not in nodes:
                    boundary_degree += adjacency_data[index]
        expansion[label] = boundary_degree / len(nodes)
    return expansion


def expansion(adjacency, labels):
    return _compute_expansion(
        adjacency.indptr,
        adjacency.indices,
        adjacency.data,
        labels.indptr,
        labels.indices,
    )


@njit
def _compute_cut_ratio(
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
    labels_indptr,
    labels_indices,
):
    cut_ratio = np.empty(len(labels_indptr) - 1)
    for label in range(len(labels_indptr) - 1):
        if labels_indptr[label] == labels_indptr[label + 1]:  # Empty cluster
            cut_ratio[label] = np.nan
            continue
        nodes = labels_indices[labels_indptr[label] : labels_indptr[label + 1]]
        boundary_degree = 0
        for node in nodes:
            neighbor_indices = np.arange(
                adjacency_indptr[node], adjacency_indptr[node + 1]
            )
            # Add in label edge weights
            for index in neighbor_indices:
                neighbor = adjacency_indices[index]
                if neighbor not in nodes:
                    boundary_degree += adjacency_data[index]
        cut_ratio[label] = boundary_degree / (
            len(nodes) * (len(adjacency_indptr) - 1 - len(nodes))
        )
    return cut_ratio


def cut_ratio(adjacency, labels):
    return _compute_cut_ratio(
        adjacency.indptr,
        adjacency.indices,
        adjacency.data,
        labels.indptr,
        labels.indices,
    )


@njit
def _compute_modularity(
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
    labels_indptr,
    labels_indices,
):
    modularities = np.empty(len(labels_indptr) - 1)
    graph_volume = np.sum(adjacency_data)
    for label in range(len(labels_indptr) - 1):
        if labels_indptr[label] == labels_indptr[label + 1]:  # Empty cluster
            modularities[label] = np.nan
            continue
        nodes = labels_indices[labels_indptr[label] : labels_indptr[label + 1]]
        internal_degree = 0
        total_degree = 0
        for node in nodes:
            neighbor_indices = np.arange(
                adjacency_indptr[node], adjacency_indptr[node + 1]
            )
            total_degree += np.sum(adjacency_data[neighbor_indices])
            # Add in label edge weights
            for index in neighbor_indices:
                neighbor = adjacency_indices[index]
                if neighbor in nodes:
                    internal_degree += adjacency_data[index]
        modularities[label] = (
            internal_degree / graph_volume - (total_degree / graph_volume) ** 2
        ) / 4
    return modularities


def modularity(adjacency, labels):
    return _compute_modularity(
        adjacency.indptr,
        adjacency.indices,
        adjacency.data,
        labels.indptr,
        labels.indices,
    )


def clustering_coefficient(adjacency, labels):
    cc = np.empty(labels.shape[0])
    for i in range(labels.shape[0]):
        nodes = labels.indices[labels.indptr[i] : labels.indptr[i + 1]]
        if len(nodes) < 3:
            cc[i] = np.nan
            continue
        label_subgraph = adjacency[nodes][:, nodes]
        cc[i] = sn.topology.get_clustering_coefficient(label_subgraph)
    return cc


def cluster_f1s(labels, predict, drop_outliers=False):
    if isinstance(labels, np.ndarray):
        labels = CAS.labels_array_to_matrix(labels)
    if isinstance(predict, np.ndarray):
        predict = CAS.labels_array_to_matrix(predict)

    if labels.shape[0] == 0 or predict.shape[0] == 0:  # No clusters
        if drop_outliers and (
            np.all(predict.getnnz(0) == 0) or np.all(predict.getnnz(0) == 0)
        ):
            return 1
        else:
            return 0

    # Drop nodes with no clusters.
    if drop_outliers:
        labels = labels.transpose().tocsr()
        label_outlier = labels.getnnz(1) == 0
        predict = predict.transpose().tocsr()
        predict_outlier = predict.getnnz(1) == 0
        outlier = np.bitwise_or(label_outlier, predict_outlier)
        labels = labels[~outlier]
        labels = labels.transpose().tocsr()
        predict = predict[~outlier]
        predict = predict.transpose().tocsr()

    label_props = np.asarray(labels.sum(axis=1)).reshape(-1)
    label_props = label_props / np.sum(label_props)
    predict_props = np.asarray(predict.sum(axis=1)).reshape(-1)
    predict_props = predict_props / np.sum(predict_props)

    overlap = predict.astype("float64") * labels.transpose().astype("float64")
    precision = overlap.multiply(1 / predict.sum(axis=1))
    recall = overlap.multiply(1 / labels.sum(axis=1).reshape(-1))
    denom = precision + recall
    denom.data = 1 / denom.data
    f1 = 2 * precision.multiply(recall).multiply(denom)
    predict_f1 = f1.max(axis=1).toarray().reshape(-1)
    predict_average = np.sum(predict_f1 * predict_props)

    labels_f1 = f1.max(axis=0).toarray().reshape(-1)
    labels_average = np.sum(labels_f1 * label_props)

    return 2 / (1 / predict_average + 1 / labels_average)
