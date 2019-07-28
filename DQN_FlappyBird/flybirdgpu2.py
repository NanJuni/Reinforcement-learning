from ple.games.flappybird import FlappyBird
from ple import PLE
import torch
import numpy as np
import DQN_agentgpu2
from PIL import Image
import time
from matplotlib import pyplot as plt
import os
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQN_agentgpu2.DQN_agent()
episode = 500000
observe_step = 2
game = FlappyBird()


p = PLE(game, fps=30, display_screen=True )
p.init()
p.reset_game()
p.act(119)
observation1 = p.getScreenRGB()
observation1 = Image.fromarray(observation1).convert('L')
observation1 = observation1.resize((120, 80))
# plt.imshow(np.array(observation1), plt.cm.gray)
# plt.show()
p.act(0)
observation2 = p.getScreenRGB()
observation2 = Image.fromarray(observation2).convert('L')
observation2 = observation2.resize((120, 80))
p.act(119)
observation3 = p.getScreenRGB()
observation3 = Image.fromarray(observation3).convert('L')
observation3 = observation3.resize((120, 80))
p.act(0)
observation4 = p.getScreenRGB()
observation4 = Image.fromarray(observation4).convert('L')
observation4 = observation4.resize((120, 80))
p.act(119)
observation5 = p.getScreenRGB()
observation5 = Image.fromarray(observation5).convert('L')
observation5 = observation5.resize((120, 80))
be_observation = np.array([[np.array(observation1), np.array(observation2), np.array(observation3), np.array(observation4)]])
cur_observation = np.array([[np.array(observation2), np.array(observation3), np.array(observation4), np.array(observation5)]])

state = (cur_observation - be_observation)/255
state = torch.cuda.FloatTensor(state, device=device)

e = 0
t = 0
R = 0
star = time.time()

while e < episode:
    if p.game_over():
        par = agent.target_net.state_dict()
        torch.save(par, "par2.pth")
        e += 1
        print("episode", e,"time_slot", t,"score", R)
        R = 0
        t = 0
        p.reset_game()
        p.act(119)
        observation1 = p.getScreenRGB()
        observation1 = Image.fromarray(observation1).convert('L')
        observation1 = observation1.resize((120, 80))
        # plt.imshow(np.array(observation1), plt.cm.gray)
        # plt.show()
        p.act(0)
        observation2 = p.getScreenRGB()
        observation2 = Image.fromarray(observation2).convert('L')
        observation2 = observation2.resize((120, 80))
        p.act(119)
        observation3 = p.getScreenRGB()
        observation3 = Image.fromarray(observation3).convert('L')
        observation3 = observation3.resize((120, 80))
        p.act(0)
        observation4 = p.getScreenRGB()
        observation4 = Image.fromarray(observation4).convert('L')
        observation4 = observation4.resize((120, 80))
        p.act(119)
        observation5 = p.getScreenRGB()
        observation5 = Image.fromarray(observation5).convert('L')
        observation5 = observation5.resize((120, 80))
        be_observation = np.array(
            [[np.array(observation1), np.array(observation2), np.array(observation3), np.array(observation4)]])
        cur_observation = np.array(
            [[np.array(observation2), np.array(observation3), np.array(observation4), np.array(observation5)]])
        state = (cur_observation - be_observation) / 255
        state = torch.cuda.FloatTensor(state, device=device)

    t += 1
    if e<observe_step:
        action = np.random.randint(0, 2, 1)
        action = int(action)
    else:
        action = agent.choose_action(state)

    if action == 1:
        reward = p.act(119)
    else:
        reward = p.act(0)

    if reward >0:
        R += 1
        reward = 10

    if reward < 0:
        terminal = 0
    else:
        terminal = 1

    observation6 = p.getScreenRGB()
    observation6 = Image.fromarray(observation6).convert('L')
    observation6 = observation6.resize((120, 80))
    ne_observation = np.array(
        [[np.array(observation3), np.array(observation4), np.array(observation5), np.array(observation6)]])
    ne_state = (ne_observation - cur_observation) / 255
    ne_state = torch.cuda.FloatTensor(ne_state, device=device)

    action = torch.cuda.LongTensor([[action]], device=device)
    reward = torch.cuda.FloatTensor([[reward]], device=device)
    terminal = torch.cuda.FloatTensor([[terminal]], device=device)
    exp = agent.experience(state, action, reward, ne_state, terminal)
    agent.save_experience(exp)
    state = ne_state
    cur_observation = ne_observation
    observation3 = observation4
    observation4 = observation5
    observation5 = observation6
    if e > observe_step:
        agent.update_train_net()
        if e % 10 == 0:
            agent.update_target_net()


