import numpy as np
from matplotlib import pyplot as plt


class HeatMap:
    def __init__(self, env, uncertainty, axis):
        self.env = env
        self.uncertainty = uncertainty
        self.axis = axis
        self.last_count = np.zeros(self.uncertainty.count.shape)

    def generate1D(self):
        self.axis.bar(self.uncertainty.bin_mids, self.uncertainty.count - self.last_count, width=self.uncertainty.bin_sizes[0], color='g')

    def reset_count(self):
        self.last_count = np.copy(self.uncertainty.count)