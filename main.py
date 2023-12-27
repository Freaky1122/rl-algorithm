import argparse
import gym
import os
import torch
import torch.multiprocessing as mp

from DQN import DQN
from utils import ReplayBuffer
from PolicyGradient import PolicyGradient
from A2C import A2C
from A3C import A3CAgent
from A3C import ActorCritic
from DDPG import DDPG
from TD3 import TD3

from PPO import PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# global_network = ActorCritic(4, 2)
# optimizer = torch.optim.Adam(global_network.parameters(), lr=0.001)
#
#
#
# def worker(input_dim, n_actions, gamma, lr):
#     env = gym.make('CartPole-v1', render_mode="human")
#     local_agent = A3CAgent(input_dim, n_actions, gamma, lr)
#     local_agent.network.to(device)  # 移动到CUDA上
#     local_agent.sync_with_global(global_network)
#     total_episode = 0
#     for _ in range(1000):  # 可以调整此数值
#         state = env.reset()
#         total_reward = 0
#         done = False
#         while not done:
#             action = local_agent.choose_action(state)
#             next_state, reward, done, _ = env.step(action)
#             local_agent.train(state, action, reward, next_state, done)
#             state = next_state
#             total_reward += reward
#         print(f"Reward: {total_reward}")
#         total_episode += 1
#         if total_episode % 10 == 0:
#             local_agent.sync_with_global(global_network)


def main():
    parser = argparse.ArgumentParser(description="Parameters for DQN Agent")
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--env", default="Pendulum-v1")
    args = parser.parse_args()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    env = gym.make(args.env)

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, 10000)
    # initialize DQN agent
    # agent = DQN(state_dim, action_dim)
    # agent = PolicyGradient(state_dim, action_dim)
    # agent = A2C(state_dim, action_dim)
    agent = TD3(state_dim, action_dim, max_action)
    max_episode = 1000

    total_reward = 0
    total_step = 0

    for i in range(max_episode):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            # agent.train(state, action, reward, next_state, done)
            # agent.rewards.append(reward)
            replay_buffer.add(state, action, next_state, reward, done)
            if replay_buffer.size > 1000:
                agent.train(replay_buffer)
            state = next_state

            total_step += 1
            total_reward += reward
        # agent.train()
        if i % 10 == 0:
            print("Episode{}:Total Step:{}, Total Reward:{}".format(i, total_step / 10, total_reward / 10))
            total_reward = 0
            total_step = 0


# def main():
#     mp.set_start_method('spawn', force=True)
#
#     processes = []
#     for _ in range(3):  # 创建与 CPU 核心数相同的进程数
#         p = mp.Process(target=worker, args=(4, 2, 0.99, 0.001))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()

# memory = []
#
#
# def main():
#     env = gym.make('CartPole-v1')
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#
#     agent = PPO(state_dim, action_dim)
#
#     for i_episode in range(10000):
#         state = env.reset()
#         done = False
#         episode_reward = 0
#         while not done:
#             action = agent.choose_action(state)
#             next_state, reward, done, _ = env.step(action)
#
#             old_prob = agent.actor(torch.tensor(state, dtype=torch.float32).to(device))[action]
#
#             memory.append((state, action, reward, next_state, done, old_prob))
#
#             episode_reward += reward
#             state = next_state
#         agent.train(memory)
#         memory.clear()
#         print(f"Episode: {i_episode}, Reward: {episode_reward}")


if __name__ == "__main__":
    main()
