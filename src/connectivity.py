import numpy as np


def fully_connected_network(n, proportion_inhib, avg_strength='auto'):
    """
    Defines a fully connected graph W with columns as inputs and rows as outputs
    To be used with left multiplication like `output = W @ input`

    Parameters
    ----------

    n : Int
        Total number of neurons in the network

    proportion_inhib: float [0,.5]
        Proportion of inhibitory neurons
        (with negative outgoing weights)

    avg_strength: float or 'auto'
        if 'auto', sets the average strength to 1/n,
        Weights are sampled uniformly (see Notes section)
        Note: It is not the average _absolute_ strength.
              Strong inhibitory neurons push the average down.

    Notes
    --------
    Our simple implementation follows this steps:
        1. Create a uniformly random n x n matrix
        2. Define proportion of the neurons to be inhibitory.
        3. Change all connections for inhibitory neurons multiplying by -1.
        4. Let average connection be avg_strength by dividing all weights.
        5. Remove self-connections
    """
    assert proportion_inhib < .5, "You must have less than half inhibitory"

    W = np.random.rand(n, n)  # Step 1

    random_inhib = (np.random.rand(n) < proportion_inhib)  # Step 2
    W[:, random_inhib] = -1 * W[:, random_inhib]  # Step 3

    # Step 4
    if avg_strength == 'auto':
        avg_strength = 1 / n

    ratio = avg_strength / W.mean()
    W = ratio * W

    # Step 5
    for i in range(n):
        W[i, i] = 0

    return W


def square_lattice_network(n, k, proportion_inhib, avg_strength='auto'):
    """
    To be used with left multiplication like `output = W @ input`

    Parameters
    ----------

    n : Int
        Total number of neurons in the network

    k : Int
        Number of nearest-neighbors connections. In this case k_in == k_out,
        the number of ingoing is equal to the number of outgoing connections.

    proportion_inhib: float [0,1]
        Proportion of inhibitory neurons
        (with negative outgoing weights)

    avg_strength: float or 'auto'
        if 'auto', sets the average strength to 1/n, by setting the nonzero average to 1/k
        Weights are sampled uniformly (see Notes section)
        Note: It is not the average _absolute_ strength.
              Strong inhibitory neurons push the average down.

    References
    --------
    [1] Lombardi, F., Herrmann, H. J., & de Arcangelis, L. (2017).
    Balance of excitation and inhibition determines 1/f power spectrum in neuronal networks.
    Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(4), 047402.
    """

    def nearest_neighbors(i):
        left = int(np.floor(k / 2))
        right = int(np.ceil(k / 2) + 1)
        neighbors = list(range(i - left, i + right))
        neighbors.remove(i)
        return np.mod(neighbors, n)

    if avg_strength == 'auto':
        avg_strength = 1 / k

    base_weights = fully_connected_network(n, proportion_inhib, avg_strength=avg_strength)

    W = np.zeros((n, n), float)

    for i in range(n):
        W[nearest_neighbors(i), i] = 1

    return W * base_weights


def small_world_network(n, k, r, proportion_inhib, avg_strength='auto'):
    """
    To be used with left multiplication like `output = W @ input`

    Parameters
    ----------

    n : Int
        Total number of neurons in the network

    k : Int
        Number of nearest-neighbors connections. In this case k_in == k_out,
        the number of ingoing is equal to the number of outgoing connections.

    r : float [0, .5]
        Proportion of rewired connections

    proportion_inhib: float [0,1]
        Proportion of inhibitory neurons
        (with negative outgoing weights)

    avg_strength: float or 'auto'
        if 'auto', sets the average strength to 1/k,
        Weights are sampled uniformly (see Notes section)
        Note: It is not the average _absolute_ strength.
              Strong inhibitory neurons push the average down.

    References
    --------
    [1] Lombardi, F., Herrmann, H. J., & de Arcangelis, L. (2017).
    Balance of excitation and inhibition determines 1/f power spectrum in neuronal networks.
    Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(4), 047402.
    """

    base_network = square_lattice_network(n, k, proportion_inhib, avg_strength)
    out_index, in_index = np.nonzero(base_network)

    rewire_index = np.random.choice(len(out_index), round(r * len(out_index)), replace=False)
    out_rewire, in_rewire = out_index[rewire_index], in_index[rewire_index]

    W = base_network.copy()
    W[(out_rewire, in_rewire)] = 0
    new_out = np.random.choice(n, len(out_rewire))
    W[(new_out, in_rewire)] = base_network[(out_rewire, in_rewire)]

    return W


def neuron_scale_free_network(n, k_out_max, proportion_inhib, avg_strength='auto'):
    """
    To be used with left multiplication like `output = W @ input`

    Parameters
    ----------

    n : Int
        Total number of neurons in the network

    k_out_max : Int
        Maximum connections. In this case k_in == k_out,
        the number of ingoing is equal to the number of outgoing connections.

    proportion_inhib: float [0,1]
        Proportion of inhibitory neurons
        (with negative outgoing weights)

    avg_strength: float or 'auto'
        if 'auto', sets the average strength to 1/k,
        Weights are sampled uniformly (see Notes section)
        Note: It is not the average _absolute_ strength.
              Strong inhibitory neurons push the average down.

    References
    --------
    [1] Lombardi, F., Herrmann, H. J., & de Arcangelis, L. (2017).
    Balance of excitation and inhibition determines 1/f power spectrum in neuronal networks.
    Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(4), 047402.
    """
    possible_ks = np.arange(2, k_out_max + 1)
    k_prob = (possible_ks**(-2.))
    k_prob = k_prob/k_prob.sum()
    k_out = np.random.choice(possible_ks, n, p=k_prob)
    if avg_strength == 'auto':
        avg_strength = 1 / k_out.mean()

    base_weights = fully_connected_network(n, proportion_inhib, avg_strength=avg_strength)
    W = np.zeros_like(base_weights)
    for i in range(n):
        possibilities = list(range(n))
        possibilities.remove(i)
        probas = 1/np.mod(np.abs(np.array(possibilities) - i), n)
        connections = np.random.choice(possibilities, k_out[i],replace=False, p=probas/probas.sum())
        W[connections, i] = base_weights[connections, i]

    return W