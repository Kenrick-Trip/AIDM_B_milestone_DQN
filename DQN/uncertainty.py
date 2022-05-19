"""
From the Deep Reinforcement Learning Course @ TU Delft
"""
import torch as th
import numpy as np


class CountUncertainty:
    """ Defines an uncertainty estimate based on counts over the state/observation space.
        Uncertainty will be scaled by 'scale'. Define boundaries either by 'state_bounds'
        or automatically by passing the environment 'env'. The counts will use
        'resolution'^m different bins for m-dimensional state vectors"""

    def __init__(self, env=None, scale=1, state_bounds=None, resolution=50, first_n_dim=None):
        if state_bounds is None:
            self.bounds = [(l, h) for l, h in zip(env.observation_space.low, env.observation_space.high)]
            if first_n_dim is not None:
                self.bounds = self.bounds[0:first_n_dim]
        else:
            self.bounds = state_bounds

        self.resolution = resolution
        self.count = th.zeros(*([resolution for _ in self.bounds]), dtype=th.long)
        self.scale = scale
        self.eps = 1E-7
        self.env = env
        self.first_n_dim = first_n_dim

        self.is_maze = "maze" in self.env.spec.id

        if not self.is_maze:
            self.calculate_bins_mc()

    def calculate_bins_mc(self):
        """Divide the state space into n bins for mc"""
        bound_sizes = np.array([b-a for a,b in self.bounds])
        self.bin_sizes = bound_sizes / self.resolution
        low_bounds = np.array([bound[0] for bound in self.bounds])
        self.bin_ends = low_bounds[0] + np.cumsum(np.full(self.resolution, self.bin_sizes[0]))
        self.bin_begins = self.bin_ends - self.bin_sizes[0]
        self.bin_mids = (self.bin_begins + self.bin_ends) / 2

    def state_bin(self, state):
        """ Find the correct bin in 'self.count' for one state. """
        cut_state = state[0][0:len(self.bounds)]
        if not self.is_maze:
            zipped = zip(cut_state, self.bounds)
            return tuple([int((x - l) / (h - l + self.eps) * self.resolution) for x, (l, h) in zipped])
        else:
            return tuple(cut_state.astype(int))

    def observe(self, state, **kwargs):
        """ Add counts for observed 'state's.
            'state' can be either a Tuple, List, 1d Tensor or 2d Tensor (1d Tensors stacked in dim=0). """

        if self.is_maze:
            state = tuple(state[0].astype(int)[:self.first_n_dim])
            self.count[state] += 1
        else:
            if isinstance(state, th.Tensor):
                if len(state.shape) == 1: state.unsqueeze_(dim=0)
            else:
                state = [state]
            for s in state:
                b = self.state_bin(s)
                self.count[b] += 1

    def get_visit_count(self, state):
        bin = self.state_bin([state])
        return self.count[bin]

    def __call__(self, state, **kwargs):
        """ Returns the estimated uncertainty for observing a (minibatch of) state(s) ans Tensor.
            'state' can be either a Tuple, List, 1d Tensor or 2d Tensor (1d Tensors stacked in dim=0).
            Does not change the counters. """
        if isinstance(state, th.Tensor):
            if len(state.shape) == 1: state.unsqueeze_(dim=0)
        else:
            state = [state]
        n = th.zeros(len(state))
        for i, s in enumerate(state):
            b = self.state_bin(s)
            n[i] = self.count[b]
        return self.scale / np.sqrt(n + self.eps)
