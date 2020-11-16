import sys
import pickle
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix
import gc
sys.path.append("./src")


from network import Network
import connectivity as conn
from itertools import product

proportion_inhib_vals = [.1, .2, .3]
synaptic_efficacies = np.linspace(.1,.8, 5).round(2)
repetitions = range(3)
omegas = np.logspace(-1, 1.2, 5).round(2)
topologies = {
    'full': lambda prop: conn.fully_connected_network(500, proportion_inhib=prop),
    'small_world': lambda prop: conn.small_world_network(500, 4, .1, proportion_inhib=prop),
    'scale_free': lambda prop: conn.neuron_scale_free_network(500, 100, proportion_inhib=prop)
}


parameter_combinations = list(product(repetitions, proportion_inhib_vals,
                                      synaptic_efficacies, omegas, topologies.items()))


for (i, proportion_inhib,
     eff, omega, (topol_name, topol_f)) in tqdm(parameter_combinations):  # ex. range(3, 30)

    W = topol_f(proportion_inhib)
    net = Network(W, synaptic_efficacy=eff, omega=omega)

    net.run_schedule(stim_durations = [   100,   100,  100,  200, 10000],
                    stim_strenghts = [.00005, .0005, .005, 0.01,  .005])

    filepath = f'data/s{i}_W{topol_name}_omg{omega}_p{proportion_inhib}_eff{eff}.pkl'
    spikes = net.get_history('s')
    pickle.dump(coo_matrix(spikes), open(filepath, 'wb'))
    gc.collect()