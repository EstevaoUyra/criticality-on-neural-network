import numpy as np
import pandas as pd


def avalanche_basic_stats(spike_matrix):
    """
    Example
    -------
    total_avalanches, avalanche_sizes, avalanche_durations = avalanche_stats(spike_matrix)
    """
    total_spikes = spike_matrix.sum(axis=0)
    total_spikes = np.hstack(([0], total_spikes))  # Assure it starts with no spike
    avalanche = pd.Series(total_spikes >= 1)

    event = (avalanche.astype(int).diff().fillna(0) != 0)
    event_indexes = event.to_numpy().nonzero()[0]

    if len(event_indexes) % 2 == 0:
        avalanche_periods = event_indexes.reshape(-1, 2)
    else:
        avalanche_periods = event_indexes[:-1].reshape(-1, 2)

    avalanche_durations = avalanche_periods[:, 1] - avalanche_periods[:, 0]

    avalanche_sizes = np.array([total_spikes[avalanche_periods[i][0]: avalanche_periods[i][1]].sum()
                                for i in range(len(avalanche_periods))])

    total_avalanches = len(avalanche_sizes)

    return total_avalanches, avalanche_sizes, avalanche_durations