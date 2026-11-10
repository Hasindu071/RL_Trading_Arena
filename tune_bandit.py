"""
tune_bandit.py
==============
Hyperparameter tuning for Phase 1 bandit agent.
Tests different window sizes, epsilon decays, and starting epsilons.
"""

import numpy as np
from bandit_env import TradingBanditEnv
from agent_base import BanditAgent


class TunableAgent(BanditAgent):
    """Testing wrapper with tunable hyperparameters."""
    
    def __init__(self, n_arms, window_size=30, epsilon_start=0.3, 
                 epsilon_decay=0.995, epsilon_min=0.05):
        super().__init__(n_arms)
        self.window_size = window_size
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.recent_rewards = [[] for _ in range(n_arms)]
        self.pull_count = 0
        self._rng = np.random.default_rng()

    def _get_current_epsilon(self):
        epsilon = self.epsilon_start * (self.epsilon_decay ** self.pull_count)
        return max(epsilon, self.epsilon_min)

    def _get_arm_value(self, arm):
        if len(self.recent_rewards[arm]) == 0:
            return 0.0
        return float(np.mean(self.recent_rewards[arm]))

    def select_arm(self) -> int:
        epsilon = self._get_current_epsilon()
        if self._rng.random() < epsilon:
            return int(self._rng.integers(0, self.n_arms))
        else:
            values = np.array([self._get_arm_value(arm) for arm in range(self.n_arms)])
            return int(np.argmax(values))

    def update(self, arm: int, reward: float) -> None:
        self.recent_rewards[arm].append(reward)
        if len(self.recent_rewards[arm]) > self.window_size:
            self.recent_rewards[arm].pop(0)
        self.pull_count += 1


def evaluate_agent(agent_class, params, n_seeds=7):
    """Evaluate agent over n_seeds replicate runs."""
    env = TradingBanditEnv(epoch_length=20, total_pulls=200)
    total_rewards = []
    
    for seed in range(n_seeds):
        env.reset(seed=seed)
        agent = agent_class(n_arms=5, **params)
        cumulative_reward = 0.0
        
        for pull in range(200):
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)
            cumulative_reward += reward
        
        total_rewards.append(cumulative_reward)
    
    return np.mean(total_rewards)


# Test configurations - comprehensive final tuning
configs = [
    {"name": "w=20, e=0.25, d=0.995, emin=0.05", 
     "params": {"window_size": 20, "epsilon_start": 0.25, "epsilon_decay": 0.995, "epsilon_min": 0.05}},
    
    {"name": "w=22, e=0.26, d=0.995, emin=0.05", 
     "params": {"window_size": 22, "epsilon_start": 0.26, "epsilon_decay": 0.995, "epsilon_min": 0.05}},
    
    {"name": "w=24, e=0.26, d=0.995, emin=0.05", 
     "params": {"window_size": 24, "epsilon_start": 0.26, "epsilon_decay": 0.995, "epsilon_min": 0.05}},
    
    {"name": "w=25, e=0.26, d=0.995, emin=0.05 (current best)", 
     "params": {"window_size": 25, "epsilon_start": 0.26, "epsilon_decay": 0.995, "epsilon_min": 0.05}},
    
    {"name": "w=26, e=0.26, d=0.995, emin=0.05", 
     "params": {"window_size": 26, "epsilon_start": 0.26, "epsilon_decay": 0.995, "epsilon_min": 0.05}},
    
    {"name": "w=28, e=0.26, d=0.995, emin=0.05", 
     "params": {"window_size": 28, "epsilon_start": 0.26, "epsilon_decay": 0.995, "epsilon_min": 0.05}},
    
    {"name": "w=25, e=0.25, d=0.995, emin=0.05", 
     "params": {"window_size": 25, "epsilon_start": 0.25, "epsilon_decay": 0.995, "epsilon_min": 0.05}},
    
    {"name": "w=25, e=0.27, d=0.995, emin=0.05", 
     "params": {"window_size": 25, "epsilon_start": 0.27, "epsilon_decay": 0.995, "epsilon_min": 0.05}},
]

print("Tuning Phase 1 Bandit Agent (7 seeds per config)\n")
print(f"{'Config':<50} {'Score':>10}")
print("-" * 62)

best_score = 0
best_config = None

for cfg in configs:
    score = evaluate_agent(TunableAgent, cfg["params"], n_seeds=7)
    print(f"{cfg['name']:<50} {score:>10.4f}")
    
    if score > best_score:
        best_score = score
        best_config = cfg

print("-" * 62)
print(f"BEST: {best_config['name']}")
print(f"Score: {best_score:.4f}")
