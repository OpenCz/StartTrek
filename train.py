#!/usr/bin/env python3

import csv
from collections import deque
from pathlib import Path

import gymnasium as gym

SEED = 0
EPSILON = 1.0
METRICS_PATH = Path("data/metrics.csv")


def make_env(seed):
	env = gym.make("LunarLander-v3", render_mode="human")
	obs, _info = env.reset(seed=seed)
	return env, obs


def init_metrics_file(path):
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle, delimiter=";")
		writer.writerow(["episode", "seed", "score", "episode_length", "termination_cause", "epsilon"])


def play_step(env, obs):
	action = env.action_space.sample()
	next_obs, reward, terminated, truncated, _info = env.step(action)
	return next_obs, reward, terminated, truncated


def get_termination_cause(obs, truncated):
	x, y = float(obs[0]), float(obs[1])
	left_leg, right_leg = float(obs[6]), float(obs[7])
	if abs(x) > 1.05 or y < -0.1 or y > 1.6 or truncated:
		return "out-of-view"
	if left_leg > 0.5 and right_leg > 0.5:
		return "sleep"
	return "crash"


def append_metric(path, episode, seed, score, episode_length, cause, epsilon):
	with path.open("a", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle, delimiter=";")
		writer.writerow([episode, seed, f"{score:.2f}", episode_length, cause, f"{epsilon:.2f}"])


def get_stats_100(recent_scores):
	if len(recent_scores) != 100:
		return
	mean_100 = sum(recent_scores) / 100
	min_100 = min(recent_scores)
	max_100 = max(recent_scores)
	print(f"stats_100 mean={mean_100:.2f} min={min_100:.2f} max={max_100:.2f}")


def finalize_episode(env, obs, recent_scores, episode, episode_score, episode_length, truncated):
	cause = get_termination_cause(obs, truncated)
	append_metric(METRICS_PATH, episode, SEED, episode_score, episode_length, cause, EPSILON)
	print(f"episode={episode} score={episode_score:.2f} length={episode_length} cause={cause}")
	recent_scores.append(episode_score)
	get_stats_100(recent_scores)
	obs, _info = env.reset()
	return obs


def run_random_loop(env, obs, recent_scores):
	episode = 0
	episode_score = 0.0
	episode_length = 0
	while True:
		obs, reward, terminated, truncated = play_step(env, obs)
		episode_score += reward
		episode_length += 1
		if terminated or truncated:
			obs = finalize_episode(env, obs, recent_scores, episode, episode_score, episode_length, truncated)
			episode += 1
			episode_score = 0.0
			episode_length = 0


def main():
	init_metrics_file(METRICS_PATH)
	env, obs = make_env(SEED)
	recent_scores = deque(maxlen=100)

	try:
		run_random_loop(env, obs, recent_scores)
	except KeyboardInterrupt:
		env.close()


if __name__ == "__main__":
	main()