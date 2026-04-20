#!/usr/bin/env python3

from collections import deque
import gymnasium as gym

SEED = 0

class randomLander:

	def __init__(self, seed):
		self.seed = seed
		self.env = gym.make("LunarLander-v3", render_mode="human")
		self.obs, _info = self.env.reset(seed=self.seed)
		self.recent_scores = deque(maxlen=100)
		self.episode = 0
		self.episode_reward = 0.0

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

		if terminated or truncated:
			print(f"Episode {self.episode} finished with reward {self.episode_reward:.2f}")
			self.recent_scores.append(self.episode_reward)
			self.display_stats()
			self.episode += 1
			self.episode_reward = 0.0
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