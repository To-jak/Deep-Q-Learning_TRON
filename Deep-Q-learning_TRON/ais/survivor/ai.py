from tron.player import Player, Direction
from tron.game import Tile, Game, PositionPlayer
from tron.window import Window
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
#from ais.basic.ai import Ai as AiBasic

import os

# General parameters
folderName = 'survivor'

# Net parameters
BATCH_SIZE = 128
GAMMA = 0.9 # Discount factor

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.batch_size = BATCH_SIZE
		self.gamma = GAMMA
		self.conv1 = nn.Conv2d(1, 32, 6)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.fc1 = nn.Linear(64*5*5, 512)
		self.fc2 = nn.Linear(512, 4)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(-1, 64*5*5)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class Ai(Player):

	def __init__(self,epsilon=0):
		super(Ai, self).__init__()
		self.net = Net()
		self.epsilon = epsilon
		# Load network weights if they have been initialized already
		if os.path.isfile('ais/' + folderName + '/ai.bak'):
			self.net.load_state_dict(torch.load('ais/' + folderName + '/ai.bak'))
			#print("load reussi 1 ")
			
		elif os.path.isfile(self.find_file('ai.bak')):
			self.net.load_state_dict(torch.load(self.find_file('ai.bak')))
			#print("load reussi 2 ")

	def action(self, map, id):

		game_map = map.state_for_player(id)

		input = np.reshape(game_map, (1, 1, game_map.shape[0], game_map.shape[1]))
		input = torch.from_numpy(input).float()
		output = self.net(input)

		_, predicted = torch.max(output.data, 1)
		predicted = predicted.numpy()
		next_action = predicted[0] + 1

		if random.random() <= self.epsilon:
			next_action = random.randint(1,4)

		if next_action == 1:
			next_direction = Direction.UP
		if next_action == 2:
			next_direction = Direction.RIGHT
		if next_action == 3:
			next_direction = Direction.DOWN
		if next_action == 4:
			next_direction = Direction.LEFT

		return next_direction

