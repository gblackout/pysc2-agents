from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.env import sc2_env
from pysc2.env import environment


class NoStopWatch(sc2_env.SC2Env):
    """
      Custom env wrapping SC2Env, to get rid of stopwatch bug(?)
    """

    def __init__(self, env):
        super(NoStopWatch, self).__init__(env)

    def reset(self):
        """Start a new episode."""
        self._episode_steps = 0
        if self._episode_count:
            # No need to restart for the first episode.
            self._controller.restart()

        self._episode_count += 1

        self._last_score = None
        self._state = environment.StepType.FIRST
        return self._step()

    def step(self, actions):
        """Apply actions, step the world forward, and return observations."""
        if self._state == environment.StepType.LAST:
            return self.reset()

        assert len(actions) == 1  # No multiplayer yet.
        action = self._features.transform_action(self._obs.observation, actions[0])
        self._controller.act(action)
        self._state = environment.StepType.MID
        return self._step()
