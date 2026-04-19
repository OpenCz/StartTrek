#!/usr/bin/env python3

import argparse
import statistics
import gymnasium as gym

SEED = 0

env = gym.make("LunarLander-v3", render_mode="human")


def policy(obs):
	del obs
	return env.action_space.sample()


obs, info = env.reset(seed=SEED)
episode = 0

while True:
	action = policy(obs)
	obs, r, terminated, truncated, info = env.step(action)
	if terminated or truncated:
		print(f"episode={episode} ended, terminated={terminated}, truncated={truncated}")
		episode += 1
		obs, info = env.reset()