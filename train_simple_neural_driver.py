import os
import pandas as pd
from collections import namedtuple
from neural_net import CarControl, FeatureTransformer

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


files = ["aalborg.csv", "alpine-1.csv", "f-speedway.csv"]
folder = "./train_data"
files = [os.path.join(folder, file) for file in files]

df = pd.read_csv(files[0])
for file in files[1:]:
	df = pd.concat((pd.read_csv(file), df))

data = df.to_dict("records")

SimpleState = namedtuple("SimpleState", data[0].keys())


class SimpleCarControl(CarControl):

	def __init__(self, network_sizes):
		super().__init__(19, network_sizes)

	def forward(self, x):
		x = Variable(x, requires_grad=True)
		i1 = F.sigmoid(self.input_layer(x))
		i2 = F.sigmoid(self.hidden_layer(i1))
		# steering is a real number 
		# TODO: should it be clamped?
		steering = self.output_steering(i2)

		# brake is in [0, 1]
		brake = F.sigmoid(self.output_brake(i2))

		# acceleration is in [0, 1] TODO verify
		acceleration = F.sigmoid(self.output_acceleration(i2))
		
		return torch.cat([steering, brake, acceleration])

class SimpleFeatureTransformer(FeatureTransformer):
	def __init__(self):
		super().__init__(exclude_from_sensor_dict={
				"ACCELERATION", "BRAKE", "STEERING", "SPEED", "TRACK_POSITION"
			}, n_history=0, size=19)

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

		## TODO NORM THE VECTOR!
		return torch.FloatTensor(feature_vector)

def eval_simple_net(net, test_data, feat_transformer):
	pass

def float_to_v(f, requires_grad=True):
	return Variable(torch.FloatTensor([f]), requires_grad=requires_grad)

def train_simple_net(net, train_data, feat_transformer, learning_rate=0.0001):
	criterion = nn.MSELoss()
	# simple SGD
	for i, d in enumerate(train_data):

		steering, brake, acceleration = net(feat_transformer.transform(d))
		
		steering_loss = criterion(steering, float_to_v(d.STEERING, requires_grad=False))		
		net.zero_grad() 
		steering_loss.backward(retain_graph=True)
		for f in net.parameters():
			f.data.sub_(f.grad.data * learning_rate)

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
			print((steering, steering_loss), brake, acceleration)

if __name__ == '__main__':
	

	#print(df.columns)

	# shuffle: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
	df = df.sample(frac=1).reset_index(drop=True)

	train_idx = int(.6 * df.shape[0])
	val_idx = train_idx + int(0.2 * df.shape[0])


	
	data = [SimpleState(**d) for d in data]

	

	train_data = data[:train_idx]
	val_data = data[train_idx: val_idx]
	test_data = data[val_idx:]


	simple_ctrl = SimpleCarControl([10, 10])
	simple_ft = SimpleFeatureTransformer()

	train_simple_net(simple_ctrl, train_data, simple_ft)

	torch.save(simple_ctrl, "models/simple_models/trail_one.pty")