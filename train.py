#!/usr/bin/env python3

from collections import deque

import gymnasium as gym

def make_env():
	env = gym.make("LunarLander-v3", render_mode="human")
	obs, _info = env.reset(seed=0)
	return env, obs

def get_stats_100(recent_scores):
	if len(recent_scores) == 100:
		mean_100 = sum(recent_scores) / 100
		min_100 = min(recent_scores)
		max_100 = max(recent_scores)
		print(f"stats_100 mean={mean_100:.2f} " f"min={min_100:.2f} max={max_100:.2f}")

def run_random_loop(env, obs, recent_scores):
	episode = 0
	episode_score = 0.0

	while True:
		action = env.action_space.sample()
		obs, reward, terminated, truncated, _info = env.step(action)
		episode_score += reward

		if terminated or truncated:
			print(f"episode={episode} total_score={episode_score:.2f}")
			recent_scores.append(episode_score)
			get_stats_100(recent_scores)
			episode += 1
			episode_score = 0.0
			obs, _info = env.reset()

def main():
	env, obs = make_env()
	recent_scores = deque(maxlen=100)

	try:
		run_random_loop(env, obs, recent_scores)
	except KeyboardInterrupt:
		env.close()


if __name__ == "__main__":
	main()