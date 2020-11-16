import numpy as np
from collections import defaultdict


class Network(object):
    def __init__(self,
                 W,
                 v0 = None,
                 omega = None,
                 synaptic_efficacy = .3,
                 history_variables=['s']):
        
        """
        Parameters
        ----------
        W : Square Array[Float] shaped (n_neurons, n_neurons)
            Connectivity matrix.

        v0 : Array[float] or None
            The initial state of the neurons.
            Defaults to all zeros

        omega : Array[float] or None
            The strength of each thalamic input.
            Defaults to a deterministic 8/n_neurons

        history_variables: list of string
            Be careful with saving a lot of things.
            It may break RAM (especially saving W)
            You can save s, v, omega, W, r, etc.
        """
        self._n_neurons = W.shape[0]

        # Set automatic values for nongiven variables
        if omega is None:
            omega = np.ones((self._n_neurons, 1)) * 8 / self._n_neurons
        elif type(omega) is not np.ndarray:
            omega = np.ones((self._n_neurons, 1)) * omega / self._n_neurons

        if v0 is None:
            v0 = np.zeros((self._n_neurons, 1))

        # Set the network state variables
        self.v = v0
        self.s = np.zeros_like(v0)
        
        self.W0 = W.copy()
        self.W = W
        self.omega0 = omega.copy()
        self.omega = omega 
        self.r = None   
        self.s_thalamus = None  # Thalamic spikes

        # Set network parameters
        self.tau_r = 400       # Recovery rate for short term depression
        self.tau_d = 20        # Decay rate for short term depression
        self.leakage = .2      # Equals 1 / membrane time constant tau m
        self.synaptic_efficacy = synaptic_efficacy

        # Additional variables
        self.history_variables = history_variables
        self.history = defaultdict(lambda: [])

    def spike(self, v):
        s = (v >= np.random.rand(*v.shape)) # Probabilistic spike
        v[s] = 0
        return s, v

    def propagate_spike(self, s):
        nt = self.W @ s
        return nt # Neurotransmitters

    def synapse(self, nt):
        dv = self.synaptic_efficacy * nt  # Delta voltage
        return dv

    def external_input(self):
        self.s_thalamus = (np.random.rand(*self.v.shape) < self.r)
        return self.s_thalamus * self.omega

    def plasticity(self):
        """
        There are two components in each update (equal for both W and omega)
            The recovery component, "homeostatic" (1/self.tau_r) * (self.W0 - self.W)
            The decay component, only when spikes occur (1/self.tau_d) * self.W * self.s 
        """
        self.W = self.W + (1/self.tau_r) * (self.W0 - self.W) - (1/self.tau_d) * self.W * self.s
        self.omega = self.omega + (1/self.tau_r) * (self.omega0 - self.omega) - (1/self.tau_d) * self.omega * self.s_thalamus

    def run_simulation(self, r, tmax):
        """
        Parameters
        ----------
        
        r: float, between 0 and 1
            Strenght of talamic input (initialized in simulation)
            It defines the probability that a given thalamic neuron will fire
            
        tmax : integer
            Number of iterations to run the network
        """
        self.r = r
        for t_i in range(tmax):
            self.s, v = self.spike(self.v)
            nt        = self.propagate_spike(self.s)
            ei        = self.external_input()
            dv        = self.synapse(nt) + ei
            self.v    = v + dv - v * self.leakage
            
            self.append_history()
            self.plasticity()

    def append_history(self):
        for var in self.history_variables:
            self.history[var].append(getattr(self, var))
            
    def get_history(self, attr):
        return np.hstack(self.history[attr])
    
    def run_schedule(self, stim_durations, stim_strenghts):
        assert len(stim_durations) == len(stim_strenghts), "Should be same size"
        for tmax, r in zip(stim_durations, stim_strenghts):
            self.run_simulation(r, tmax)