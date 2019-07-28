import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from collections import namedtuple
import random
import os
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DQN_net(nn.Module):
    def __init__(self):
        super(DQN_net, self).__init__()
        self.c1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=2, padding=2)
        self.p1 = nn.MaxPool2d(kernel_size=2)
        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        # self.p2 = nn.MaxPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.p3 = nn.MaxPool2d(kernel_size=2)
        self.d1 = nn.Linear(8064, 256)
        self.d2 = nn.Linear(256, 256)
        self.d3 = nn.Linear(256, 2)

    def forward(self, input):
        x = self.c1(input)
        x = self.p1(x)
        x = self.c2(x)
        x = F.relu(x)
        # x = self.p2(x)
        x = self.c3(x)
        x = F.relu(x)
        # x = self.p3(x)
        # x = torch.flatten(x, start_dim=0, end_dim=-1)
        x = x.view(x.size(0), -1)
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        output = self.d3(x)
        return output

class DQN_agent():
    def __init__(self):
        super(DQN_agent, self).__init__()
        self.lr = 0.000001
        self.dr = 0.99
        self.epsilon = 0.1
        self.batch_size = 64
        self.train_net = DQN_net().to(device)
        self.target_net = DQN_net().to(device)
        self.target_net.load_state_dict(self.train_net.state_dict())
        self.opt = Adam(self.train_net.parameters(), self.lr)
        self.loss = nn.MSELoss()
        self.experience = namedtuple("agent_experience", ['state', 'action', 'reward', 'next_state', 'terminal'])
        self.experience_pool = []


    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            action_Q_value = self.train_net.forward(state)
            action = action_Q_value.argmax().item()
        else:
            action = np.random.randint(0, 2, 1)
        return int(action)

    def save_experience(self, exp):
        if len(self.experience_pool) >= 30000:
            self.experience_pool = self.experience_pool[1: ]  # 保持经验池经验个数恒定
        self.experience_pool.append(exp)

    def update_target_net(self):
        self.target_net.load_state_dict(self.train_net.state_dict())

    def update_train_net(self):
        if len(self.experience_pool) > self.batch_size:
            batch_experience = random.choices(self.experience_pool, k=self.batch_size)   # 对经验池进行批处理
            batch_experience = self.experience(*zip(*batch_experience))
            batch_state = torch.cat(batch_experience.state)
            batch_action = torch.cat(batch_experience.action)
            batch_reward = torch.cat(batch_experience.reward)
            batch_next_state = torch.cat(batch_experience.next_state)
            batct_terminal = torch.cat(batch_experience.terminal)
            batch_train_Q = self.train_net.forward(batch_state).gather(dim=1, index=batch_action)
            batch_target_Q = self.dr * self.target_net.forward(batch_next_state).detach().max(dim=1)[0].view(self.batch_size, 1)*batct_terminal + batch_reward
            batch_loss = self.loss(batch_train_Q, batch_target_Q)
            # batch_loss = batch_loss / self.batch_size
            self.opt.zero_grad()
            batch_loss.backward()
            self.opt.step()
        else:
            pass

