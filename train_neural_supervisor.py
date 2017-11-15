from my_driver import MyDriver
from pytocl.driver import Driver

from run_torcs import run
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

from neural_net import FeatureTransformer, CarControl

import numpy as np

import sys

class SupervisedDriver(Driver):

	def __init__(self, trainee, feature_transformer, persist_path, optimizer_params=None, loss=None):
		super(SupervisedDriver, self).__init__()
		self.supervisor = MyDriver()
		self.feature_transformer = feature_transformer
		self.trainee = trainee
		self.persist_path = persist_path

		if optimizer_params is None:
			optimizer_params = {
				"lr" : 0.001,
				"momentum": 0.9
			}

		if loss is None:
			loss = nn.MSELoss()

		# loss
		self.loss = loss

		# optimizer 
		self.optimizer = optim.SGD(self.trainee.parameters(), **optimizer_params)

		self.call_number = 0

	def drive(self, state):
		self.call_number += 1

		command = self.supervisor.drive(state)
		
		x = self.feature_transformer.transform(state)
		y_predicted = self.trainee(x)
		y_true = Variable(torch.FloatTensor([command.steering, command.brake, command.accelerator]))

		self.optimizer.zero_grad()
		loss = self.loss(y_predicted, y_true)		
		loss.backward()

		if not np.isnan(loss.data.numpy()):
			self.optimizer.step()

		if self.call_number % 1000 == 0:
			print("Step: {}".format(self.call_number))
			print("Prediction", y_predicted)
			print("True", y_true)
			print("loss", loss)

			print()

		return command


	def on_shutdown(self):
		torch.save(self.trainee, self.persist_path)

if __name__ == '__main__':

	# python <file> config train/retrain persist_path
	if len(sys.argv) != 4:
		raise ValueError()

	feature_transformer = FeatureTransformer()

	if sys.argv[2] == "retrain":
		trainee = torch.load(sys.argv[3])
	else:
		trainee = CarControl(feature_transformer.size, [200, 50, 10])

	run(SupervisedDriver(trainee, feature_transformer, sys.argv[3]), sys.argv[1])
