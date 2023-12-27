import torch
import torch.nn as nn
import torch.optim as optim
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value


class A3CAgent:
    def __init__(self,  input_dim, n_actions, gamma, lr):
        self.network = ActorCritic(input_dim, n_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, _ = self.network(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device).view(-1, 1)

        _, value = self.network(state)
        _, next_value = self.network(next_state)
        expected_value = reward + self.gamma * next_value * (1 - int(done))
        td_error = expected_value - value

        probs, _ = self.network(state)
        log_prob = torch.log(probs.gather(1, action))
        actor_loss = -log_prob * td_error.detach()

        critic_loss = td_error**2

        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        self.optimizer.step()

    def sync_with_global(self, global_network):
        self.network.load_state_dict(global_network.state_dict())
