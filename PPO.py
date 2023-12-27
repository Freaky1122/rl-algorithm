import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(self, input_dim, n_actions, gamma=0.99, lr=0.001, clip_epsilon=0.2, update_iterations=2):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_iterations = update_iterations

        self.actor = Actor(input_dim, n_actions).to(device)
        self.critic = Critic(input_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        probs = self.actor(state)
        action = np.random.choice(len(probs.detach().cpu().numpy()), p=probs.detach().cpu().numpy())
        return action

    def train(self, memory):
        for _ in range(self.update_iterations):
            for state, action, reward, next_state, done, old_prob in memory:
                state = torch.tensor(state, dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
                action = torch.tensor(action).to(device)
                old_prob = torch.tensor(old_prob).to(device)

                prob = self.actor(state)
                prob = prob[action]

                critic_value = self.critic(state)
                advantage = reward + self.gamma * self.critic(next_state) - critic_value

                ratio = prob / old_prob
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage

                actor_loss = -torch.min(surr1, surr2)
                critic_loss = advantage ** 2

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                actor_loss.backward(retain_graph=True)
                critic_loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()
