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

    def state_bin(self, state):
        """ Find the correct bin in 'self.count' for one state. """
        zipped = zip(state[0][0:len(self.bounds)], self.bounds)
        return tuple([int((x - l) / (h - l + self.eps) * self.resolution) for x, (l, h) in zipped])

    def observe(self, state, **kwargs):
        """ Add counts for observed 'state's.
            'state' can be either a Tuple, List, 1d Tensor or 2d Tensor (1d Tensors stacked in dim=0). """
        if isinstance(state, th.Tensor):
            if len(state.shape) == 1: state.unsqueeze_(dim=0)
        else:
            state = [state]
        for s in state:
            b = self.state_bin(s)
            self.count[b] += 1

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