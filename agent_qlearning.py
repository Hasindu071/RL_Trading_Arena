"""
agent_qlearning.py  --  Phase 2 (MDP) submission: tabular Q-learning
====================================================================
Off-policy TD control. The update target uses the BEST next action:

    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

Improvements over the untuned QLearningReferenceAgent:
  * high starting epsilon with per-episode decay (explores early, exploits late)
  * longer discount horizon (gamma = 0.99)
  * per-(state, action) learning rate that decays with visit count
  * greedy during evaluation (set_eval_mode)
  * random tie-breaking in argmax

Run:  python3 test_locally.py agent_qlearning.py mdp
Submit as agent.py (the server looks for a class named exactly `Agent`).
"""

import numpy as np
from agent_base import MDPAgent


class Agent(MDPAgent):
    def __init__(self, n_states, n_actions,
                 gamma=0.99,
                 alpha_start=0.3, alpha_min=0.05, alpha_decay_visits=50,
                 eps_start=1.0, eps_min=0.05, eps_decay=0.995):
        super().__init__(n_states, n_actions)
        self.gamma = gamma
        self.alpha_start = alpha_start
        self.alpha_min = alpha_min
        self.alpha_decay_visits = alpha_decay_visits

        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.epsilon = eps_start

        self.Q = np.zeros((n_states, n_actions))
        self.visits = np.zeros((n_states, n_actions))
        self._episode = 0
        self._is_eval = False
        self._rng = np.random.default_rng()

    # -- helpers ------------------------------------------------------- #
    def _greedy(self, state):
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self._rng.choice(best))

    def _alpha(self, state, action):
        # starts at alpha_start, shrinks toward alpha_min as (s,a) is visited
        a = self.alpha_start / (1.0 + self.visits[state, action] / self.alpha_decay_visits)
        return max(self.alpha_min, a)

    # -- MDPAgent interface ------------------------------------------- #
    def start_episode(self):
        # Decay epsilon once per episode. Do NOT reset Q here.
        if self._episode > 0:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        self._episode += 1

    def set_eval_mode(self, is_eval):
        self._is_eval = is_eval

    def act(self, state):
        if not self._is_eval and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return self._greedy(state)

    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])   # <-- Q-learning: max
        alpha = self._alpha(state, action)
        self.Q[state, action] += alpha * (target - self.Q[state, action])
        self.visits[state, action] += 1
