"""
agent.py  --  Phase 1 (bandit) example submission
=================================================
Contains several bandit algorithms. Only ONE `Agent` class may be active
(uncommented) at a time -- the harness looks for a class named exactly `Agent`.

To switch algorithms:
  1. Comment out the currently active class (put `#` in front of each line).
  2. Uncomment the one you want to try.
  3. Run:  python3 test_locally.py agent.py bandit

Active by default: Sliding-Window UCB (best suited to the non-stationary
market in bandit_env.py, since the best arm changes between phases).

Tip: in an editor, select a block and press Ctrl+/ (Cmd+/ on Mac) to
toggle comments on all selected lines at once.
"""

import numpy as np
from agent_base import BanditAgent


# ===================================================================== #
# 0. RANDOM (provided baseline)  -- uniform random arm every pull
# ===================================================================== #
class Agent(BanditAgent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self._rng = np.random.default_rng()

    def select_arm(self) -> int:
        return int(self._rng.integers(0, self.n_arms))

    def update(self, arm: int, reward: float) -> None:
        pass


# ===================================================================== #
# 1. EPSILON-GREEDY (with decay, but never below a floor)
#    Explore with probability epsilon, otherwise pick the best-looking arm.
#    Uses a constant step size (alpha) instead of 1/n so old rewards fade --
#    this helps with non-stationarity. The epsilon floor keeps exploring.
# ===================================================================== #
# class Agent(BanditAgent):
#     def __init__(self, n_arms, eps_start=0.3, eps_min=0.05, eps_decay=0.98, alpha=0.5):
#         super().__init__(n_arms)
#         self.epsilon = eps_start
#         self.eps_min = eps_min
#         self.eps_decay = eps_decay
#         self.alpha = alpha                  # constant step size -> forgets old data
#         self.values = np.zeros(n_arms)
#         self.counts = np.zeros(n_arms)
#         self._rng = np.random.default_rng()

#     def select_arm(self) -> int:
#         if self._rng.random() < self.epsilon:
#             arm = int(self._rng.integers(0, self.n_arms))
#         else:
#             # break ties randomly so we don't always favour arm 0
#             best = np.flatnonzero(self.values == self.values.max())
#             arm = int(self._rng.choice(best))
#         self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
#         return arm

#     def update(self, arm: int, reward: float) -> None:
#         self.counts[arm] += 1
#         self.values[arm] += self.alpha * (reward - self.values[arm])


# ===================================================================== #
# 2. UCB1 (classic, stationary)
#    Picks the arm with the highest  mean + c * sqrt(ln(t) / n_arm).
#    Rewards here are small (weekly returns ~ +-0.05), so the exploration
#    constant `c` is scaled down to match the reward magnitude.
#    NOTE: assumes stationary arms, so it will lag when the market shifts.
# ===================================================================== #
# class Agent(BanditAgent):
#     def __init__(self, n_arms, c=0.05, alpha=0.50):
#         super().__init__(n_arms)
#         self.c = c
#         self.alpha = alpha
#         self.counts = np.zeros(n_arms)
#         self.values = np.zeros(n_arms)
#         self.t = 0

#     def select_arm(self) -> int:
#         untried = np.where(self.counts == 0)[0]

#         if len(untried) > 0:
#             return int(untried[0])

#         bonus = self.c * np.sqrt(
#             np.log(self.t) / self.counts
#         )

#         return int(np.argmax(self.values + bonus))

#     def update(self, arm: int, reward: float) -> None:
#         self.t += 1
#         self.counts[arm] += 1

#         # Constant step-size update
#         self.values[arm] += self.alpha * (
#             reward - self.values[arm]
#         )

# ===================================================================== #
# 3. THOMPSON SAMPLING (Gaussian posterior, with discounting)
#    Keeps a Normal posterior over each arm's mean reward, samples from
#    each, and plays the arm with the highest sample. Discounting
#    (gamma < 1) shrinks old evidence each step so the posterior can
#    re-widen after the market changes.
# ===================================================================== #
# class Agent(BanditAgent):
#     def __init__(self, n_arms, gamma=0.95, obs_sigma=0.05, prior_sigma=0.15):
#         super().__init__(n_arms)
#         self.gamma = gamma                  # 1.0 = no forgetting (stationary TS)
#         self.obs_var = obs_sigma ** 2       # assumed reward noise variance
#         self.prior_prec = 1.0 / prior_sigma ** 2
#         self.n_eff = np.zeros(n_arms)       # discounted pull counts
#         self.sum_eff = np.zeros(n_arms)     # discounted reward sums
#         self._rng = np.random.default_rng()

#     def select_arm(self) -> int:
#         prec = self.prior_prec + self.n_eff / self.obs_var
#         mean = (self.sum_eff / self.obs_var) / prec     # prior mean = 0
#         samples = self._rng.normal(mean, 1.0 / np.sqrt(prec))
#         return int(np.argmax(samples))

#     def update(self, arm: int, reward: float) -> None:
#         self.n_eff *= self.gamma            # fade all arms' evidence
#         self.sum_eff *= self.gamma
#         self.n_eff[arm] += 1
#         self.sum_eff[arm] += reward


# ===================================================================== #
# 4. SLIDING-WINDOW UCB  (ACTIVE)
#    UCB computed only over the last `window` pulls. Evidence older than
#    that is dropped completely, so when the market moves from bull ->
#    choppy -> bear, stale estimates disappear and arms whose recent
#    samples left the window get re-explored via their large bonus.
#    Window ~ 40-60 works well since each market phase lasts ~66 pulls.
# ===================================================================== #
# class Agent(BanditAgent):
#     def __init__(self, n_arms, window=30, c=0.05):
#         super().__init__(n_arms)
#         self.window = window
#         self.c = c                          # scaled to weekly-return magnitude
#         self.history = []                   # list of (arm, reward), newest last

#     def select_arm(self) -> int:
#         recent = self.history[-self.window:]
#         counts = np.zeros(self.n_arms)
#         sums = np.zeros(self.n_arms)
#         for a, r in recent:
#             counts[a] += 1
#             sums[a] += r

#         # any arm absent from the window gets (re)tried first
#         unseen = np.flatnonzero(counts == 0)
#         if len(unseen) > 0:
#             return int(unseen[0])

#         t = len(recent)
#         means = sums / counts
#         bonus = self.c * np.sqrt(np.log(t) / counts)
#         return int(np.argmax(means + bonus))

#     def update(self, arm: int, reward: float) -> None:
#         self.history.append((arm, reward))