import torch
from torch import nn 
import torch.nn.functional as F
from torch.autograd import Variable

from pytocl.car import State

class FeatureTransformer:

	

	def __init__(self, exclude_from_sensor_dict=None, n_history=10, size=35):
		self.previous_states = []
		self.n_history = n_history
		self.size = size

		if not exclude_from_sensor_dict:
			exclude_from_sensor_dict = {"current_lap_time",
				"last_lap_time", "opponents", "race_position", "wheel_velocities", "distance_raced"}

		self.exclude_from_sensor_dict = exclude_from_sensor_dict

	def transform(self, state):
		sensor_dict = vars(state)
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

		#self.previous_states.append(state)
		#if len(self.previous_states) > self.n_history:
		#	self.previous_states.pop(0)




class CarControl(nn.Module):

	def __init__(self, n_inputs, linear_sizes):
		super(CarControl, self).__init__()

		print("Number of inputs: {}".format(n_inputs))
		assert len(linear_sizes) == 2
		self.input_layer = nn.Linear(n_inputs, linear_sizes[0]) 
		self.hidden_layer = nn.Linear(linear_sizes[0], linear_sizes[1])
	
		self.output_steering = nn.Linear(linear_sizes[1], 1)
		self.output_acceleration = nn.Linear(linear_sizes[1], 1)
		self.output_brake = nn.Linear(linear_sizes[1], 1)

	def forward(self, x):
		x = Variable(x, requires_grad=True)
		i1 = F.sigmoid(self.input_layer(x))
		i2 = F.sigmoid(self.hidden_layer(i1))
		# steering is a real number 
		# TODO: should it be clamped?
		steering = torch.clamp(self.output_steering(i2), min=-1, max=1)

		# brake is in [0, 1]
		brake = torch.clamp(F.sigmoid(self.output_brake(i2)), min=0, max=0.25)

		# acceleration is in [0, 1] TODO verify
		acceleration = torch.clamp(self.output_acceleration(i2), min=0.1, max=1)
		
		return torch.cat([steering, brake, acceleration])


if __name__ == '__main__':
	

	s = State({'angle': '0.008838', 'wheelSpinVel': ['67.9393', '68.8267', '71.4009', '71.7363'],
			'rpm': '4509.31', 'focus': ['26.0077', '27.9798', '30.2855', '33.0162', '36.3006'],
			'trackPos': '0.126012', 'fuel': '93.9356', 'speedX': '81.5135', 'speedZ': '-2.4422',
			'track': ['4.3701', '4.52608', '5.02757', '6.07753', '8.25773', '11.1429', '13.451',
			'16.712', '21.5022', '30.2855', '51.8667', '185.376', '69.9077', '26.6353',
			'12.6621', '8.2019', '6.5479', '5.82979', '5.63029'], 'gear': '3',
			'damage': '0', 'distRaced': '42.6238', 'z': '0.336726', 'racePos': '1',
			'speedY': '0.40771', 'curLapTime': '4.052', 'lastLapTime': '0',
			'distFromStart': '1015.56',
			'opponents': ['123.4', '200', '200', '200', '200', '200', '200', '200', '200', '200',
			 '200', '200', '200', '200', '200', '200', '200', '200', '200', '200',
			 '200', '200', '200', '200', '200', '200', '200', '200', '200', '200',
			 '200', '200', '200', '200', '200', '200']})

	feature_transformer = FeatureTransformer()
	feature_transformer.transform(s)

	steering_control = CarControl(feature_transformer.size, [50, 50])
	print(steering_control)
	print(steering_control(feature_transformer.transform(s)))


	n_mutations = 10
	weight = steering_control.input_layer.weight.clone()
	
	for mutation_no in range(n_mutations):
		random_noise = torch.rand(weight.size())
		weight.add_(Variable(random_noise))
		print(weight)




