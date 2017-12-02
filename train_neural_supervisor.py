import os
import sys

import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

from my_driver import MyDriver
from neural_net import FeatureTransformer, CarControl
from pytocl.driver import Driver
from run_torcs import run


class SupervisedDriver(Driver):
    def __init__(self, trainee, feature_transformer, persist_path, optimizer_params=None, loss=None):
        super(SupervisedDriver, self).__init__()
        self.supervisor = MyDriver()
        self.feature_transformer = feature_transformer
        self.trainee = trainee
        self.persist_path = persist_path

        if optimizer_params is None:
            optimizer_params = {
                "lr": 0.001,
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

    # python <file> train/retrain persist_path iterations
    if len(sys.argv) != 4:
        raise ValueError()

    mode = sys.argv[1]
    model_path = sys.argv[2]
    number_of_iterations = sys.argv[3]

    feature_transformer = FeatureTransformer()

    if mode == "retrain":
        trainee = torch.load(model_path)
    else:
        trainee = CarControl(feature_transformer.size, [200, 50, 10])

    config_folder = "./config_files/"
    for iteration in range(1, int(number_of_iterations) + 1):
        print("Train Iteration", iteration)
        for path in os.listdir(config_folder):
            config_path = os.path.join(config_folder, path)
            config_path = os.path.abspath(config_path)
            print("Config Path: {}".format(config_path))
            run(SupervisedDriver(trainee, feature_transformer, model_path), config_path)
