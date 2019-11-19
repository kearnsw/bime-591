from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import warnings
import random
import gym

# Details at https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# State: 
# Num	Observation                 Min         Max
# 0		Cart Position              -4.8         4.8
# 1		Cart Velocity              -Inf         Inf
# 2		Pole Angle                 -24 deg      24 deg
# 3		Pole Velocity At Tip       -Inf         Inf

# Actions:
# Type: Discrete(2)
# Num	Action
# 0		Push cart to the left
# 1		Push cart to the right

# Handcrafted Agent
class handcrafted_agent():
	def choose_action(self, state):
		return 0

# Deep Q Network Agent
class dqn_agent():
	def __init__(self):
		self.memory = []
		self.environment = gym.make('CartPole-v1')
		self.epsilon = 1
		self.epsilon_decay = 0.995
		self.epsilon_min = 0.01
		self.n_win = 150
		self.batch_size = 32

		# Deep Q Network
		self.model = Sequential()
		self.model.add(Dense(24, input_dim = 4, activation = 'tanh'))
		self.model.add(Dense(48, activation = 'tanh'))
		self.model.add(Dense(2))
		self.model.compile(loss = 'mse', optimizer=Adam(lr = 0.1, decay = 0.05))

	# Given a state, choose an action. Pick the action with the highest expected
	# Q value.  
	def choose_action(self, state):
		if (np.random.random() <= self.epsilon):
			return self.environment.action_space.sample()
		else:
			return np.argmax(self.model.predict(state))


	# At the end of each episode, update weights based on a batch of random
	# moves from previous episodes. 
	def update_weights(self):
		x, y = [], []

		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		for state, action, reward, next_state, done in batch:
			y_target = self.model.predict(state)
			if done:
				y_target[0][action] = reward 
			else:
				y_target[0][action] = reward + np.max(self.model.predict(next_state)[0])
			x.append(state[0])
			y.append(y_target[0])
			
		self.model.fit(np.array(x), np.array(y),  verbose = 0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	# Play through a single episode. 
	def play_episode(self):

		i = 0
		done = False
		state = self.environment.reset().reshape(1,4)
		while not done:
			action = self.choose_action(state)
			next_state, reward, done, info = self.environment.step(action)
			next_state = next_state.reshape(1,4)
			self.memory.append((state, action, reward, next_state, done))
			state = next_state
			i += 1

		return i

	# Play through many episodes and update weights after each one. 
	def train(self, episodes):
		scores = []
		for episode in range(episodes):
			score = self.play_episode()
			scores.append(score)

			if episode % 100 == 0:
				print(f'[Episode {episode}] - You can last {np.mean(scores[-100:])} moves.')
			if episode >= 100:
				if np.mean(scores[-100:]) > self.n_win:
					print(f'Ran {episode} episodes. Solved after {episode - 100} trials ✔')
					break
				
			self.update_weights()

def main():
	environment = gym.make('CartPole-v1')
	state = environment.reset()
	for i in range(200):
		environment.render()
		# Take a random action.
		state, reward, done, info = environment.step(environment.action_space.sample())
	environment.close()

	tf.logging.set_verbosity(tf.logging.ERROR)
	agent = dqn_agent()

	agent.train(1000)

	environment = gym.make('CartPole-v1')
	state = environment.reset()
	done = False
	while not done:
		environment.render()
		# Agent picks an action. 
		action = agent.choose_action(state.reshape(1,4))
		state, reward, done, info = environment.step(action)
	environment.close()

	agent = handcrafted_agent()
	environment = gym.make('CartPole-v1')
	state = environment.reset()
	done = False
	while not done:
		environment.render()
		# Agent picks an action. 
		action = agent.choose_action(state)
		state, reward, done, info = environment.step(action)
	environment.close()

if __name__ == '__main__':
	main()
