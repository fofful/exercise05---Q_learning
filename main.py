# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy as np
import time

# Environment
env = gym.make("Taxi-v3").env

alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode

# Q tables for rewards
Q_reward = np.zeros([500, 6])

# Training w/ random sampling of actions
tot_reward = 0
previous_reward = 0
previous_action = 0
done = 0
state = env.reset()
for i in range(0, num_of_episodes):
    previous_reward = 0
    previous_action = 0
    done = 0
    state = env.reset()
    for j in range(0, num_of_steps):
        previous_state = state
        action = random.randrange(0, 6)
        state, reward, done, info = env.step(action)
        Q_reward[previous_state][action] = Q_reward[previous_state][action] + alpha * (reward + gamma * np.max(Q_reward[state]) - Q_reward[previous_state][action])
        
    print(i)
tot_rewards = []
tot_actions = []
for tests in range(0, 10):
    state = env.reset()
    tot_reward = 0
    for t in range(0, 50):
        action = np.argmax(Q_reward[state])
        state, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        print(tot_reward)
        time.sleep(0.1)
        if (done == 1):
            tot_rewards.append(tot_reward)
            tot_actions.append(t)
            break

print('average total rewards: ', np.sum(tot_rewards) / len(tot_rewards))
print('average number of actions: ', np.sum(tot_actions) / len(tot_actions))
