from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class Milestone(ABC):
    """
    Represent a milestone with a reward and a goal state.
    The goal state may contain np.nan values, which means that that dimension has to be ignored for comparison.
    This can be used when you want to e.g. only look at the velocity for MountainCar and don't care about the position.
    """
    reward: float
    goal_state: NDArray

    def __post_init__(self):
        self.not_nan = ~np.isnan(self.goal_state)

    @abstractmethod
    def is_achieved(self, state: NDArray) -> bool:
        pass


class ExactMilestone(Milestone):
    """
    Represent a milestone for achieving an exact location.
    Works better for discrete spaces, such as a maze.
    """

    def is_achieved(self, state: NDArray) -> bool:
        return np.allclose(self.goal_state[self.not_nan], state[self.not_nan])


class PassingMilestone(Milestone):
    """
    Represent a milestone that has to be passed.
    Works better for continuous spaces, such as MountainCar.
    """

    def is_achieved(self, state: NDArray) -> bool:
        return (state[self.not_nan] >= self.goal_state[self.not_nan]).all()
