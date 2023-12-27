import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=128):
        super(Actor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.gamma = gamma
        self.tau = tau

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.actor(state).cpu().detach().numpy().flatten()

    def train(self, reply_buffer, batch_size=64):
        state, action, next_state, reward, done = reply_buffer.sample(batch_size)

        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        # compute Critic loss
        current_Q = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # optimize thr actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the parameters of target network
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)