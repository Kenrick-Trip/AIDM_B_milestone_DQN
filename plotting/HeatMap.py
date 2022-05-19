import numpy as np
from matplotlib import pyplot as plt


class HeatMap:
    def __init__(self, env, uncertainty, axis):
        self.env = env
        self.uncertainty = uncertainty
        self.axis = axis
        self.last_count = np.zeros(self.uncertainty.count.shape)

    def generate1D(self):
        if "MountainCar" not in self.env.spec.id:
            raise Exception("1d heatmap only supported for MountainCar")
        data = self.uncertainty.count - self.last_count
        data = data / max(data)
        self.axis.bar(self.uncertainty.bin_mids, data, width=self.uncertainty.bin_sizes[0], color='g')
        self.axis.set_xlabel("Position")

    def generate2D(self):
        if "maze" not in self.env.spec.id:
            raise Exception("2d heatmap only supported for maze")
        data = self.uncertainty.count - self.last_count
        self.axis.imshow(data.T, interpolation='nearest', cmap="summer")

    def reset_count(self):
        self.last_count = np.copy(self.uncertainty.count)