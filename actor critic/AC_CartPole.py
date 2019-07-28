import gym
import os
import numpy as np
import actorCritic
import matplotlib.pyplot as plt
import torch

env = gym.make('CartPole-v0')
env = env.unwrapped

max_episode = 2000
Render = True

save_time = np.zeros(max_episode)
agent = actorCritic.AC()
for episode in range(max_episode):
    state = env.reset()
    time = 0
    while True:
        time += 1
        action = agent.select_action(torch.tensor(state).float())
        next_state, reward, done, [] = env.step(action)
        agent.save_reward.append(reward)
        # if Render: env.render()
        if done or time >= 1000:
            break
        state = next_state
    save_time[episode] = time
    print(episode, time)
    agent.update()
    agent.init_save()
env.close()
np.savetxt("AC_01.txt", save_time)