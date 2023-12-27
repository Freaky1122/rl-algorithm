import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        action_mean = self.model(state)
        return action_mean


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.model(state)
        return value


# 定义Actor-Critic算法
class A2C:
    def __init__(self, input_dim, action_dim, hidden_dim=64, lr_actor=1e-2, lr_critic=1e-2, gamma=0.99):
        self.actor = Actor(input_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(input_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.actor(state)
        action = np.random.choice(np.array([0, 1]), p=probs.detach().cpu().numpy()[0])
        return action

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = torch.tensor([[action]], dtype=torch.int64).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        values = self.critic(state)
        next_values = self.critic(next_state)

        # TD error
        target = reward + self.gamma * next_values * (1 - done)
        td_error = target - values
        probs = self.actor(state)
        actor_loss = -torch.log(probs.gather(1, action)) * td_error.detach()
        critic_loss = F.mse_loss(values, target.detach())

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
