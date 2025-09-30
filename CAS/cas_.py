import numpy as np
import scipy.sparse as sp
from numba import njit, prange, typeof
from numba_stats import binom
from numba.typed import List
from numba.types import int32, int64


#################
# CAS Functions #
#################


@njit
def _get_degree(node, adjacency_indptr, adjacency_indices, adjacency_data):
    return np.sum(adjacency_data[adjacency_indptr[node] : adjacency_indptr[node + 1]])


@njit
def _get_degree_and_label_degree(
    node,
    label,
    node_labels,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
):
    degree = 0
    label_degree = 0
    for neighbor_index in range(adjacency_indptr[node], adjacency_indptr[node + 1]):
        neighbor = adjacency_indices[neighbor_index]
        edge_weight = adjacency_data[neighbor_index]
        degree += edge_weight
        if label in node_labels[neighbor]:
            label_degree += edge_weight
    return degree, label_degree


@njit
def _ief(
    node,
    label,
    label_volume,
    node_labels,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
):
    degree, in_label_degree = _get_degree_and_label_degree(
        node,
        label,
        node_labels,
        adjacency_indptr,
        adjacency_indices,
        adjacency_data,
    )
    if degree == 0:
        return 0
    return in_label_degree / degree


@njit
def _nief(
    node,
    label,
    label_volume,
    node_labels,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
):
    ief = _ief(
        node,
        label,
        label_volume,
        node_labels,
        adjacency_indptr,
        adjacency_indices,
        adjacency_data,
    )
    return max(ief - label_volume, 0)


@njit
def _p(
    node,
    label,
    label_volume,
    node_labels,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
):
    degree = np.empty(1, dtype="float64")
    label_degree = np.empty(1, dtype="float64")
    degree[0], label_degree[0] = _get_degree_and_label_degree(
        node,
        label,
        node_labels,
        adjacency_indptr,
        adjacency_indices,
        adjacency_data,
    )
    if degree[0] == 0:
        return 0
    p = binom._cdf(label_degree, degree, label_volume)[0]
    return p


########################
# Formatting Functions #
########################


@njit
def _labels_array_to_matrix(labels):
    n_labels = np.max(labels) + 1
    n_nonoutliers = np.sum(labels >= 0)
    indptr = np.empty(n_labels + 1, dtype="int32")
    indices = np.empty(n_nonoutliers, dtype="int32")
    data = np.ones(n_nonoutliers, dtype="bool")
    labels_argsort = np.argsort(labels)
    next_index = 0
    for label in range(n_labels):
        indptr[label] = next_index
        label_nodes = np.arange(len(labels))[labels == label]
        n_label_nodes = len(label_nodes)
        indices[next_index : next_index + n_label_nodes] = label_nodes
        next_index += n_label_nodes
    indptr[-1] = next_index  # update last indptr entry
    return indptr, indices, data


def labels_array_to_matrix(labels):
    """Convert a numpy labels array to a (labels x nodes) csr matrix
    Labels are assumed to be contiguous 0-max(labels), and -1 denotes nodes with no labels
    """
    labels_indptr, labels_indices, labels_data = _labels_array_to_matrix(labels)
    labels = sp.csr_matrix(
        (labels_data, labels_indices, labels_indptr),
        shape=(len(labels_indptr) - 1, len(labels)),
        dtype="bool",
    )
    labels.data[:] = True  # Sometime some entries are flipped to false, don't know why.
    return labels


@njit
def _make_node_label_sets(labels_indptr, labels_indices, n_nodes):
    """Take (label x node) csc matrix and return a list membership sets."""
    node_labels = List([{int32(-1)} for _ in range(n_nodes)])
    for node in range(len(labels_indptr) - 1):
        node_labels[node] = set(
            labels_indices[labels_indptr[node] : labels_indptr[node + 1]]
        )
    return node_labels


@njit
def _make_sparse_from_label_sets(node_labels, n_labels):
    """Take a list membership sets and return (labels x nodes) csc matrix."""
    n_indices = sum([len(labels) for labels in node_labels])
    indptr = np.empty(len(node_labels) + 1, dtype="int32")
    indices = np.empty(n_indices, dtype="int32")
    next_empty_index = 0
    for node in range(len(node_labels)):
        indptr[node] = next_empty_index
        this_indices = np.sort(list(node_labels[node]))
        indices[next_empty_index : next_empty_index + len(this_indices)] = this_indices
        next_empty_index += len(this_indices)
    indptr[-1] = next_empty_index
    return indptr, indices


#########################
# Post Processing Logic #
#########################


@njit
def _get_new_labels(
    node_labels,
    n_labels,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
    graph_volume,
    threshold,
    cas,
    only_remove=False,
):
    new_labels = List([{int32(-1)} for _ in range(len(node_labels))])

    # Cache label volumes
    label_volumes = np.zeros(n_labels, dtype="float32")
    for node, labels in enumerate(node_labels):
        degree = _get_degree(node, adjacency_indptr, adjacency_indices, adjacency_data)
        for label in labels:
            label_volumes[label] += degree
    assert graph_volume > 0
    label_volumes /= graph_volume

    # Compute new memberships
    for node in range(len(node_labels)):
        new_labels[node].remove(int32(-1))  # Get rid of placeholder
        plausible_labels = node_labels[node].copy()
        if not only_remove:  # Also look at neighbor's labels
            neighbors = adjacency_indices[
                adjacency_indptr[node] : adjacency_indptr[node + 1]
            ]
            for neighbor in neighbors:
                plausible_labels.update(node_labels[neighbor])
        for label in plausible_labels:
            if (
                cas(
                    node,
                    label,
                    label_volumes[label],
                    node_labels,
                    adjacency_indptr,
                    adjacency_indices,
                    adjacency_data,
                )
                >= threshold
            ):
                new_labels[node].add(label)
    return new_labels


@njit
def _post_process(
    labels_indptr,
    labels_indices,
    n_labels,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
    threshold,
    max_rounds,
    cas,
    only_remove=False,
    verbose=False,
):
    """Remove nodes from clusters if their cas score is below the threshold and add nodes
    to clusters if their cas is above the threshold. Nodes are added or removed in
    the order of greatest cas distance to the threshold.

    Parameters:
    -----------
    labels_indptr: indptr of (labels x nodes) csc matrix
    labels_indices: indices of (labels x nodes) csc matrix
    adjacency_indptr: indptr of csr adjacency matrix
    adjacency_indices: indices of csr adjacency matrix
    adjacency_data: data of csr adjacency matrix
    threshold (float)
    max_rounds: maximum rounds to recompute CAS scores
    cas: pointer to numba.njit cas function
    only_remove: Flag to only_remove nodes from their labels vs. add to new labels.
    verbose: Flag to print loop information for debugging
    """
    graph_volume = np.sum(adjacency_data)
    node_labels = _make_node_label_sets(
        labels_indptr, labels_indices, len(adjacency_indptr) - 1
    )
    nodes_that_moved_last_round = {int64(-1)}
    for round_number in range(max_rounds):
        new_labels = _get_new_labels(
            node_labels,
            n_labels,
            adjacency_indptr,
            adjacency_indices,
            adjacency_data,
            graph_volume,
            threshold,
            cas,
            only_remove=only_remove,
        )
        # Get a set of the nodes that moved
        nodes_that_moved_this_round = {int64(-1)}
        nodes_that_moved_this_round.remove(int64(-1))
        for node in range(len(node_labels)):
            if new_labels[node] != node_labels[node]:
                nodes_that_moved_this_round.add(node)
        if verbose:
            print(
                f"Round {round_number}: {len(nodes_that_moved_this_round)} nodes moved."
            )

        if len(nodes_that_moved_this_round) == 0:
            break

        # We can get stuck in a loop moving a few nodes back and forth.
        # Not the best solution but maybe fast-ish
        if (
            not only_remove
            and nodes_that_moved_this_round == nodes_that_moved_last_round
        ):
            break

        node_labels = new_labels
        nodes_that_moved_last_round = nodes_that_moved_this_round

    labels_indptr, labels_indices = _make_sparse_from_label_sets(node_labels, n_labels)
    labels_data = np.ones(len(labels_indices), dtype="bool")
    return labels_indptr, labels_indices, labels_data


class CASPostProcesser:
    """
    Post-processing community detection results to allow for outliers and/or overlapping communities.

    Parameters:
    -----------

    score: string (optional, default "nief")
        The CAS score to use. Must be one of "ief", "nief" or "p".

    threshold: float (optional, default 0.5)
        The CAS threshold for adding/removing nodes from a community.

    max_rounds: int (optional, default 100)
        The maximum number of rounds to re-compute CAS scores. Results usually
        converge in less than 20 rounds.

    only_remove: bool (optional, defualt True)
        Flag to only allow for removing nodes from their community. If True and
        starting from a partition, there will be no overlapping communities. If
        False, both add and remove nodes to communities, which may create overlap.

    sparse_output: bool (optional, default False)
        Force labels_ to be stored/returned in a labels x nodes sparse matrix for
        compatibility with overlapping communities. If True, only_remove = True, and
        a 1d numpy array is passed, labels_ are stored and returned as a 1d numpy array.

    relabel_clusters: bool (optional default True)
        Relabel clusters to contiguous range 0-n by removing empty clusters.

    verbose: bool (optional defualt False)
        Option to print details

    """

    def __init__(
        self,
        score="nief",
        threshold=0.5,
        max_rounds=100,
        only_remove=True,
        sparse_output=False,
        relabel_clusters=True,
        verbose=False,
    ):
        self.score = score
        self.threshold = threshold
        self.max_rounds = max_rounds
        self.only_remove = only_remove
        self.sparse_output = sparse_output
        self.relabel_clusters = relabel_clusters
        self.verbose = verbose

    def _validate_parameters(self):
        if self.score == "ief":
            self.cas = _ief
        elif self.score == "nief":
            self.cas = _nief
        elif self.score == "p":
            self.cas = _p
        else:
            raise ValueError(f"score must be in ['ief', 'nief', 'p']")

        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0")

        if not isinstance(self.max_rounds, int):
            if self.max_rounds % 1 != 0:
                raise ValueError("max_rounds must be a whole number")
            try:
                # convert other types of int to python int
                self.max_rounds = int(self.max_rounds)
            except ValueError:
                raise ValueError("max_rounds must be an int")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")

        if not isinstance(self.only_remove, bool):
            raise ValueError("only_remove must be a bool")

        if not isinstance(self.sparse_output, bool):
            raise ValueError("sparse_output must be a bool")

    def fit(self, labels, adjacency):
        """
        Post-process a clustering.

        Parameters:
        -----------
        labels: 1d numpy array or (labels x nodes) scipy sparse matrix

        adjacency: scipy sparse adjacency matrix. Stored values are
            interpreted as edge weights.
        """
        self._validate_parameters()
        return_as_numpy = False
        if isinstance(labels, np.ndarray):
            if labels.ndim != 1:
                raise ValueError(f"Expected 1d numpy array. Got {labels.ndim} dims.")
            labels = labels_array_to_matrix(labels).tocsc()
            return_as_numpy = (
                self.only_remove and not self.sparse_output
            )  # If passed a numpy array and only removing, return a numpy array
        elif sp.issparse(labels):
            labels = labels.tocsc()
        else:
            raise ValueError(
                f"Expected labels to be a scipy sparse array of numpy array. Got {type(labels)}"
            )

        if sp.issparse(adjacency):
            adjacency = adjacency.tocsr()
        else:
            raise ValueError(
                f"Adjacnecy must be a sparse matrix. Got {type(adjacency)}."
            )

        assert labels.format == "csc"  # labels is a (labels x nodes) csc matrix
        n_labels = labels.shape[0]
        labels_indptr, labels_indices, labels_data = _post_process(
            labels.indptr,
            labels.indices,
            n_labels,
            adjacency.indptr,
            adjacency.indices,
            adjacency.data,
            self.threshold,
            self.max_rounds,
            self.cas,
            self.only_remove,
            self.verbose,
        )  # (node x label) matrix

        labels = sp.csc_matrix(
            (labels_data, labels_indices, labels_indptr),
            shape=(n_labels, adjacency.shape[0]),
            dtype="bool",
        )
        labels.data[:] = (
            True  # Sometime some entries are flipped to false, don't know why.
        )
        labels = labels.tocsr()

        if self.relabel_clusters:
            non_empty_cluster = labels.getnnz(1) > 0
            self.old_cluster_ids = np.arange(labels.shape[0])[non_empty_cluster]
            labels = labels[non_empty_cluster]

        if return_as_numpy:
            labels = labels.tocsc()
            if not all(labels.getnnz(axis=0) <= 1):  # Check max one cluster per node
                raise AssertionError(
                    "Cannot return as numpy array, some nodes have more than one label."
                )
                self.labels_ = labels.tocsr()  # Save matrix anyway
            numpy_labels = np.full(adjacency.shape[0], -1, dtype="int32")
            numpy_labels[labels.getnnz(axis=0) == 1] = labels.indices
            labels = numpy_labels

        self.labels_ = labels
        return self

    def fit_predict(self, labels, adjacency):
        self.fit(labels, adjacency)
        return self.labels_
