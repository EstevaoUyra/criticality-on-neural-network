import numpy as np


def fully_connected_network(n, proportion_inhib, avg_strength='auto'):
    """
    Parameters
    ----------

    n : Int
        Total number of neurons in the network

    proportion_inhib: float [0,1]
        Proportion of inhibitory neurons
        (with negative outgoing weights)

    avg_strength: float or 'auto'
        if 'auto', sets the average strength to 1/n
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
