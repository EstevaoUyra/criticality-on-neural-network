import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_three_conn(func, s_mult=1, l_mult=1, **kwargs):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), dpi=100)
    for i in range(3):
        W = func(**kwargs)
        sns.heatmap(W, ax=ax[1, i])
        plot_circle(W, s_mult=s_mult, l_mult=l_mult, ax=ax[0, i])


def plot_circle(W, s_mult=1, l_mult=1, ax=None, place_random=False):
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    i = np.arange(W.shape[0])

    if place_random == False:
        x = np.sin(2 * np.pi * i / i.max())
        y = np.cos(2 * np.pi * i / i.max())
    else:
        x = np.random.rand(W.shape[0])
        y = np.random.rand(W.shape[0])

    plt.scatter(x, y, s=W.sum(axis=1) * s_mult, color='k')

    for i_to, i_from in np.vstack(np.nonzero(W)).T:
        from_xy = (x[i_from], y[i_from])
        dxy = (x[i_to] - x[i_from], y[i_to] - y[i_from])
        plt.arrow(*from_xy, *dxy, linewidth=np.abs(W[i_to, i_from]) * l_mult, color='r' if W[i_to, i_from] < 0 else 'k')
    ax.axis('off')


def loglogdensity(events, log_bins=True, bins=10, ax=None):
    if ax is None:
        ax = plt.gca()
        plt.sca(ax)

    if log_bins:
        density, bin = np.histogram(np.log10(events), density=False, bins=bins)
        durations = 10 ** bin[:-1]
    else:
        density, bin = np.histogram(events, density=False, bins=bins)

    plt.plot(durations, density, marker='o')
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Log-log plot of avalanche durations");