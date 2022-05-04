from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update


class AdaptiveDQN(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fixed_exploration_rate = 1

    def _on_step(self):
        """Overwrite _on_step method from DQN class"""
        super()._on_step()
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        # Set exploration rate to a fixed 1 for testing
        # self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.exploration_rate = 1

        self.logger.record("rollout/exploration_rate", self.exploration_rate)
