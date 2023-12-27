import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import torch.optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        return self.l2(x)


class DQN(object):
    def __init__(self, state_dim, action_dim, target_update=10, gamma=0.98, epsilon=0.99, learning_rate=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.QNet = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_QNet = copy.deepcopy(self.QNet)

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=self.lr)
        self.count = 0

    def choose_action(self, state):
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.QNet(state).argmax().item()
        return action

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, terminated = replay_buffer.sample(batch_size)

        next_max_q = self.target_QNet(next_state).max(1)[0].view(-1, 1)
        target_q = reward + self.gamma * next_max_q * (1 - terminated)
        print("targetQ:{}".format(target_q))
        predict_q = self.QNet(state).gather(1, action)[:, 0].view(-1, 1)
        print("predictQ:{}".format(predict_q))
        loss = F.mse_loss(predict_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % 10 == 0:
            self.target_QNet.load_state_dict(self.QNet.state_dict())

    def save(self, filename):
        torch.save(self.QNet.state_dict(), filename + "_QNetwork")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.QNet.load_state_dict(torch.load(filename + "_QNetwork"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
