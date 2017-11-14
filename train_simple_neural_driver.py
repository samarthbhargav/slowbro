import os
import pandas as pd
from collections import namedtuple
from neural_net import CarControl, FeatureTransformer
import numpy as np


import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import torch.optim as optim

files = ["aalborg.csv", "alpine-1.csv", "f-speedway.csv"]
folder = "./train_data"
files = [os.path.join(folder, file) for file in files]

df = pd.read_csv(files[0], index_col=False)
print(df.STEERING.mean())

for file in files[1:]:
	df1 = pd.read_csv(file, index_col=False)
	df = pd.concat((df1, df))

data = df.to_dict("records")

SimpleState = namedtuple("SimpleState", data[0].keys())


class SimpleCarControl(nn.Module):

	def __init__(self, network_sizes):
		super(SimpleCarControl, self).__init__()

		print("Number of inputs: {}".format(19))
		assert len(network_sizes) == 2
		self.input_layer = nn.Linear(19, network_sizes[0]) 
		self.hidden_layer = nn.Linear(network_sizes[0], network_sizes[1])
	
		self.output = nn.Linear(network_sizes[1], 3)
		#self.output_acceleration = nn.Linear(network_sizes[1], 1)
		#self.output_brake = nn.Linear(network_sizes[1], 1)

	def forward(self, x):
		x = Variable(x, requires_grad=True)
		i1 = F.sigmoid(self.input_layer(x))
		i2 = F.sigmoid(self.hidden_layer(i1))
		
		return self.output(i2)

class SimpleFeatureTransformer(FeatureTransformer):
	def __init__(self):
		super().__init__(exclude_from_sensor_dict={
				"ACCELERATION", "BRAKE", "STEERING", "SPEED", "TRACK_POSITION"
			}, n_history=0, size=19)

	def _normalized(self, a, axis=-1, order=2):
		a = np.array(a)
		l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
		l2[l2==0] = 1
		return a / np.expand_dims(l2, axis)

	def transform(self, state):
		if isinstance(state, SimpleState):
			sensor_dict = state._asdict()
		else:
			sensor_dict = {
				"ANGLE_TO_TRACK_AXIS" : state.angle
			}

			for i in range(18):
				sensor_dict["TRACK_EDGE_{}".format(i)] = state.distances_from_edge[i]
	
		sensors = list(sensor_dict.keys() - self.exclude_from_sensor_dict)
		sensors.sort(key=lambda _: _[0])

		feature_vector = []
		for k in sensors:
			v = sensor_dict[k]
			if isinstance(v, tuple):
				feature_vector.extend(v)
			else:
				feature_vector.append(v)

		 
		return torch.FloatTensor(feature_vector)

def eval_simple_net(net, test_data, feat_transformer):
	pass

def float_to_v(f, requires_grad=True):
	return Variable(torch.FloatTensor([f]), requires_grad=requires_grad)

def train_simple_net(net, train_data, feat_transformer, learning_rate=0.0001):
	criterion = nn.MSELoss()

	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

	# simple SGD
	for i, d in enumerate(train_data):
		inp = feat_transformer.transform(d)
		prediction = net(inp)
		optimizer.zero_grad()
		true = Variable(torch.FloatTensor([d.STEERING, d.BRAKE, d.ACCELERATION]))
		loss = criterion(prediction, true)		
		loss.backward()

		if np.isnan(loss.data.numpy()):
			continue

		optimizer.step()

		# brake_loss = criterion(brake, float_to_v(d.BRAKE, requires_grad=False))		
		# net.zero_grad() 
		# brake_loss.backward(retain_graph=True)
		# for f in net.parameters():
		# 	f.data.sub_(f.grad.data * learning_rate)

		# acceleration_loss = criterion(acceleration, float_to_v(d.ACCELERATION, requires_grad=False))		
		# net.zero_grad() 
		# acceleration_loss.backward()
		# for f in net.parameters():
		# 	f.data.sub_(f.grad.data * learning_rate)




		if i % 100 == 0:
			print("Step: {}".format(i))
			print(inp)
			print("Prediction", prediction)
			print("True", true)
			print("loss", loss)

			print()
			print()

if __name__ == '__main__':
	

	#print(df.columns)

	# shuffle: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
	df = df.sample(frac=1).reset_index(drop=True)
	train_idx = int(.9 * df.shape[0])
	val_idx = train_idx + int(0.2 * df.shape[0])


	
	data = [SimpleState(**d) for d in data]

	

	train_data = data[:train_idx]
	val_data = data[train_idx: val_idx]
	test_data = data[val_idx:]


	simple_ctrl = SimpleCarControl([10, 10])
	simple_ft = SimpleFeatureTransformer()

	train_simple_net(simple_ctrl, train_data, simple_ft)

	torch.save(simple_ctrl, "models/simple_models/trail_one.pty")

	torch.load("models/simple_models/trail_one.pty")