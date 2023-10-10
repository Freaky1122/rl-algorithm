import argparse
import gym
import os

import torch

from DQN import DQN
from utils import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Parameters for DQN Agent")
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--env", default="CartPole-v1")
    args = parser.parse_args()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    env = gym.make(args.env, render_mode="human")

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    print(action_dim)
    print(state_dim)
    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, 10000)
    # initialize DQN agent
    agent = DQN(state_dim, action_dim)
    max_episode = 50000

    total_reward = 0
    total_step = 0

    for i in range(max_episode):
        state = env.reset()
        print(state)
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            if replay_buffer.size > 300:
                agent.train(replay_buffer)
            state = next_state

            total_step += 1
            total_reward += reward
        if i % 10 == 0:
            print("Episode{}:Total Step:{}, Total Reward:{}".format(i, total_step / 10, total_reward / 10))
            total_reward = 0
            total_step = 0


if __name__ == "__main__":
    main()
