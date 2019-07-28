import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import sgd
from torch.optim import adam
from torch.distributions import Categorical
import numpy as np


state_space = 4
action_space = 2
discount_rate = 0.99
learning_rate = 0.01
eps = np.finfo(np.float32).eps.item()


class ActorCricic(nn.Module):
    def __init__(self):
        super(ActorCricic, self).__init__()
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3_a = nn.Linear(32, action_space)
        self.fc3_c = nn.Linear(32, 1)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        hidden = F.relu(self.fc2(hidden))
        output_a = F.softmax(self.fc3_a(hidden), dim=-1)
        output_c = self.fc3_c(hidden)
        return output_a, output_c


class AC:
    def __init__(self):
        super(AC, self).__init__()
        self.ac = ActorCricic()
        # self.ac_optimizer = sgd.SGD(self.ac.parameters(), lr=learning_rate)
        self.ac_optimizer = torch.optim.Adam(self.ac.parameters(), lr=learning_rate)
        # self.ac_optimizer = torch.optim.SGD(self.ac.parameters(), lr=learning_rate)
        self.save_log_prob = []
        self.save_state_vale = []
        self.save_reward = []

    def init_save(self):
        del self.save_log_prob[:]
        del self.save_state_vale[:]
        del self.save_reward[:]

    def select_action(self, state):
        action_prob, state_value = self.ac.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        self.save_log_prob.append(m.log_prob(action))
        self.save_state_vale.append(state_value)
        return action.item()

    def update(self):
        actor_loss = []
        critic_loss = []
        R = 0
        cumulate_reward = []
        for r in self.save_reward[::-1]:
            R = r + discount_rate*R
            cumulate_reward.insert(0, R)

        cumulate_reward = torch.tensor(cumulate_reward)
        cumulate_reward = (cumulate_reward - cumulate_reward.mean()) / (cumulate_reward + eps)

        for log_prob, value, reward in zip(self.save_log_prob, self.save_state_vale, cumulate_reward):
            critic_loss.append(F.smooth_l1_loss(value, torch.tensor(reward)))
            actor_loss.append(-log_prob*(reward-value.detach()))

        self.ac_optimizer.zero_grad()
        loss = torch.stack(critic_loss).sum() + torch.stack(actor_loss).sum()
        loss.backward()
        self.ac_optimizer.step()








