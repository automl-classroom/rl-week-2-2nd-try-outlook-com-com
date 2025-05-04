from __future__ import annotations

from typing import Any,SupportsFloat

import gymnasium as gym
import numpy as np

# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
        transition_probabilities: np.ndarray = np.ones((2, 2)),
        rewards: list[float] = [0,1],
        horizon: int = 10,
        seed: int | None = None):

        """Initializes the observation and action space for the environment."""
        self.rng = np.random.default_rng(seed)
        self.rewards = list(rewards)
        self.P = np.array(transition_probabilities)
        self.horizon = int(horizon)
        self.current_steps = 0
        self.position = 0 #start at 0 

        #spaces 
        n = self.P.shape[0]
        self.observation_space = gym.spaces.Discrete(n)
        self.action_space = gym.spaces.Discrete(2)

        #helpers
        self.states = np.arange(n)
        self.actions = np.arange(2)

        #transition matrix
        # self.transition_matrix = self.T = self.get_transition_matrix()
        
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        
        self.current_steps = 0
        self.position = 0
        return self.position, {}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
    
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        # stochastic flip with prob 1 - P[pos, action]
        p = float(self.P[self.position, action])
        follow = self.rng.random() < p
        a_used = action if follow else 1 - action

        delta = -1 if a_used == 0 else 1
        self.position = max(0, min(self.states[-1], self.position + delta))

        reward = float(self.rewards[self.position])
        terminated = False
        truncated = self.current_steps >= self.horizon

        return self.position, reward, terminated, truncated, {}

    
    def get_reward_per_action(self) -> np.ndarray:
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        for s in range(nS):
            for a in range(nA):
                nxt = max(0, min(nS - 1, s + (-1 if a == 0 else 1)))
                R[s, a] = float(self.rewards[nxt])
        return R

    def get_transition_matrix(
        self,
        S: np.ndarray | None = None,
        A: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        
        if S is None or A is None or P is None:
            S, A, P = self.states, self.actions, self.P

        nS, nA = len(S), len(A)
        T = np.zeros((nS, nA, nS), dtype=float)
        for s in S:
            for a in A:
                s_next = max(0, min(nS - 1, s + (-1 if a == 0 else 1)))
                T[s, a, s_next] = float(P[s, a])
        return T


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, base_env: gym.Env, observation_noise: float = 0.1, seed: int | None = None):
        super().__init__(base_env)
        assert 0.0 <= observation_noise <= 1.0, "Noise must be between 0 and 1"
        self.observation_noise = observation_noise
        self.random_generator = np.random.default_rng(seed)

        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        true_observation, reset_info = self.env.reset(seed=seed, options=options)
        return self._apply_observation_noise(true_observation), reset_info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        true_observation, reward, terminated, truncated, step_info = self.env.step(action)
        return self._apply_observation_noise(true_observation), reward, terminated, truncated, step_info

    def _apply_observation_noise(self, true_observation: int) -> int:
        if self.random_generator.random() < self.observation_noise:
            num_states = self.observation_space.n
            possible_observations = [s for s in range(num_states) if s != true_observation]
            return int(self.random_generator.choice(possible_observations))
        else:
            return int(true_observation)

    def render(self, mode: str = "human"):
        return self.env.render(mode=mode)

