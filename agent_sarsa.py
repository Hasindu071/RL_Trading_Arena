"""
agent_sarsa.py  --  Phase 2 (MDP) submission: tabular SARSA
===========================================================
On-policy TD control. The update target uses the action the agent will
ACTUALLY take next (chosen by the epsilon-greedy policy):

    Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))

The harness only calls update(s, a, r, s', done), so SARSA has to pick a'
itself. We do that inside update(): choose a' with the current epsilon-greedy
policy, use it in the target, and cache it so the very next act(s') returns
that same a'. (Otherwise the update and the real behaviour would disagree,
and it would not be true SARSA.)

Same improvements as the Q-learning file (epsilon decay, gamma = 0.99,
decaying per-(s,a) learning rate, greedy at evaluation) so the two can be
compared fairly in your report.

Run:  python3 test_locally.py agent_sarsa.py mdp
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
        self._next_action = None      # a' already chosen inside update()
        self._rng = np.random.default_rng()

    # -- helpers ------------------------------------------------------- #
    def _greedy(self, state):
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self._rng.choice(best))

    def _epsilon_greedy(self, state):
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return self._greedy(state)

    def _alpha(self, state, action):
        a = self.alpha_start / (1.0 + self.visits[state, action] / self.alpha_decay_visits)
        return max(self.alpha_min, a)

    # -- MDPAgent interface ------------------------------------------- #
    def start_episode(self):
        self._next_action = None      # never carry an action across episodes
        if self._episode > 0:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        self._episode += 1

    def set_eval_mode(self, is_eval):
        self._is_eval = is_eval
        self._next_action = None      # drop any cached training action

    def act(self, state):
        if self._is_eval:
            return self._greedy(state)
        if self._next_action is not None:
            a, self._next_action = self._next_action, None
            return a
        return self._epsilon_greedy(state)

    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            self._next_action = self._epsilon_greedy(next_state)
            target += self.gamma * self.Q[next_state, self._next_action]   # <-- SARSA: a'
        else:
            self._next_action = None
        alpha = self._alpha(state, action)
        self.Q[state, action] += alpha * (target - self.Q[state, action])
        self.visits[state, action] += 1
