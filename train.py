#!/usr/bin/env python3

import csv
from collections import deque
from pathlib import Path
import gymnasium as gym

SEED = 0
EPSILON = 1.0
METRICS_PATH = Path("data/metrics.csv")

class saveCsvLander:

	def __init__(self, path):
		self.path = Path(path)
		self.path.parent.mkdir(parents=True, exist_ok=True)
		with self.path.open("w", newline="", encoding="utf-8") as handle:
			writer = csv.writer(handle, delimiter=";")
			writer.writerow(["episode", "seed", "score", "episode_length", "termination_cause", "epsilon"])

	def get_end_cause(self, obs, truncated):
		x, y = float(obs[0]), float(obs[1])
		left_leg, right_leg = float(obs[6]), float(obs[7])
		if abs(x) > 1.05 or y < -0.1 or y > 1.6 or truncated:
			return "out-of-view"
		if left_leg > 0.5 and right_leg > 0.5:
			return "sleep"
		return "crash"

	def write_info(self, episode, seed, score, episode_length, cause, epsilon):
		with self.path.open("a", newline="", encoding="utf-8") as handle:
			writer = csv.writer(handle, delimiter=";")
			writer.writerow([episode, seed, f"{score:.2f}", episode_length, cause, f"{epsilon:.2f}"])

class randomLander:

	def __init__(self, seed):
		self.seed = seed
		self.env = gym.make("LunarLander-v3", render_mode="human")
		self.obs, _info = self.env.reset(seed=self.seed)
		self.recent_scores = deque(maxlen=100)
		self.episode = 0
		self.episode_reward = 0.0
		self.episode_length = 0
		self.csv_logger = saveCsvLander(METRICS_PATH)

	def display_stats(self):
		if len(self.recent_scores) != 100:
			return
		mean_100 = sum(self.recent_scores) / 100
		min_100 = min(self.recent_scores)
		max_100 = max(self.recent_scores)
		print(f"stats_100 mean={mean_100:.2f} min={min_100:.2f} max={max_100:.2f}")

	def step(self):
		action = self.env.action_space.sample()
		self.obs, reward, terminated, truncated, _info = self.env.step(action)
		self.episode_reward += reward
		self.episode_length += 1

		if terminated or truncated:
			cause = self.csv_logger.get_end_cause(self.obs, truncated)
			self.csv_logger.write_info(self.episode, self.seed, self.episode_reward, self.episode_length, cause, EPSILON)
			print(f"Episode {self.episode} finished with reward {self.episode_reward:.2f}")
			self.recent_scores.append(self.episode_reward)
			self.display_stats()
			self.episode += 1
			self.episode_reward = 0.0
			self.episode_length = 0
			self.obs, _info = self.env.reset()

	def run(self):
		while True:
			self.step()

	def close(self):
		self.env.close()

def main():
	runner = randomLander(seed=SEED)

	try:
		runner.run()
	except KeyboardInterrupt:
		runner.close()


if __name__ == "__main__":
	main()