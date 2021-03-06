import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from pytocl.car import State


class FeatureTransformer:
    def __init__(self, exclude_from_sensor_dict=None, n_history=10, size=31):
        self.previous_states = []
        self.n_history = n_history
        self.size = size

        if not exclude_from_sensor_dict:
            exclude_from_sensor_dict = {"current_lap_time",
                                        "last_lap_time", "opponents", "race_position", "wheel_velocities",
                                        "distance_raced", "distance_from_start", "fuel", "gear", "rpm"}

        self.exclude_from_sensor_dict = exclude_from_sensor_dict

    def transform(self, state):
        sensor_dict = vars(state)
        sensors = list(sensor_dict.keys() - self.exclude_from_sensor_dict)
        sensors.sort(key=lambda _: _[0])
        self.add_state_to_history(sensors)

        feature_vector = []
        for s in self.previous_states:
            for k in s:
                v = sensor_dict[k]
                if isinstance(v, tuple):
                    feature_vector.extend(v)
                else:
                    feature_vector.append(v)

        ## TODO NORM THE VECTOR!
        return torch.FloatTensor(feature_vector)

    # self.previous_states.append(state)
    # if len(self.previous_states) > self.n_history:
    #	self.previous_states.pop(0)

    def add_state_to_history(self, state):
        self.previous_states.insert(0, state)
        self.previous_states = self.previous_states[:self.n_history]
        while len(self.previous_states) < self.n_history:
            self.previous_states.insert(0, self.previous_states[0])


class CarControl(nn.Module):
    def __init__(self, n_inputs, layer_sizes):
        super(CarControl, self).__init__()

        print("Number of inputs: {}".format(n_inputs))
        self.hidden_layer_1 = nn.Linear(n_inputs * 10, layer_sizes[0])
        self.hidden_layer_2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.hidden_layer_3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.output = nn.Linear(layer_sizes[-1], 3)

    def forward(self, x):
        x = Variable(x, requires_grad=True)

        intermediate_1 = F.sigmoid(self.hidden_layer_1(x))
        intermediate_2 = F.sigmoid(self.hidden_layer_2(intermediate_1))
        intermediate_3 = F.sigmoid(self.hidden_layer_3(intermediate_2))

        return self.output(intermediate_3)


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
