import matplotlib
import numpy as np
from matplotlib import pyplot as plt


class HeatMap:
    def __init__(self, env, uncertainty, fig, axis):
        self.env = env
        self.uncertainty = uncertainty
        self.fig = fig
        self.axis = axis
        self.last_count = np.zeros(self.uncertainty.count.shape, dtype=int)
        self.colorbar = None

    def generate1D(self):
        if "MountainCar" not in self.env.spec.id:
            raise Exception("1d heatmap only supported for MountainCar")
        data = self.uncertainty.count - self.last_count
        data = data / max(data)
        self.axis.bar(self.uncertainty.bin_mids, data, width=self.uncertainty.bin_sizes[0], color='g')
        self.axis.set_xlabel("Position")

    def generate2D(self):
        if self.colorbar is not None:
            self.colorbar.remove()
        if "maze" not in self.env.spec.id:
            raise Exception("2d heatmap only supported for maze")
        data = (self.uncertainty.count - self.last_count).numpy().T
        plot = self.axis.imshow(data,  cmap='summer')
        self.axis.set_title("Location visitation count")
        self.axis.set_xlabel("x-position")
        self.axis.set_ylabel("y-position")
        self.colorbar = self.fig.colorbar(plot, ax=self.axis, shrink=0.7, orientation='horizontal')
        self.axis.clear()

    def reset_count(self):
        self.last_count = np.copy(self.uncertainty.count)
