import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        return F.softmax(self.l2(x), dim=1)


class PolicyGradient(object):
    def __init__(self, action_dim, state_dim, gamma=0.99, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = learning_rate

        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.saved_log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs = self.policy_net(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def train(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        print("yes")
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []
