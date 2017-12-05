import sys
import math
import torch

from neural_net import CarControl, FeatureTransformer
from simple_neural_driver import *


class NeuralDriver(Driver):
    def __init__(self, feature_transformer, network, life_span):
        super().__init__()

        self.feature_transformer = feature_transformer
        self.network = network
        self.frame = 0
        self.last_state = None
        self.karma = 0
        self.life_span = life_span * 500

    def drive(self, car_state: State) -> Command:
        """
        Produces driving command in response to newly received car state.
        """
        # print(car_state)
        # print()
        feature_vector = self.feature_transformer.transform(car_state)
        steering, brake, acceleration = self.network(feature_vector).data
        # print("steering", steering)
        # print("brake", brake)
        # print("acceleration", acceleration)

        command = Command()
        command.accelerator = acceleration
        command.brake = brake
        command.steering = steering

        # TODO make handling the gear a part of the neural net
        # if car_state.gear == 0:
        #     command.gear = 1
        # elif car_state.rpm > 8000:
        #         command.gear = car_state.gear + 1
        # elif car_state.rpm < 2500 and car_state.gear > 1:
        #     command.gear = car_state.gear - 1

        if acceleration > 0:
            if car_state.rpm > 8000:
                command.gear = car_state.gear + 1

        if car_state.rpm < 2500 and car_state.gear > 2:
            command.gear = car_state.gear - 1

        if not command.gear:
            command.gear = car_state.gear or 1

        if self.data_logger:
            self.data_logger.log(car_state, command)

        if self.frame % 1000 == 0:
            # print(command)
            # print(car_state)
            # print()
            ...

        self.last_state = car_state

        if self.frame % 10 == 0:
            self.calculate_karma()

        self.frame = (1 + self.frame) % 100000

        return command

    def calculate_karma(self):
        car_state = self.last_state
        self.karma += car_state.distance_from_start
        if math.fabs(car_state.distance_from_center) > 0.99:
            self.karma -= 500
        else:
            self.karma += car_state.speed_x * (math.cos(car_state.angle) - math.fabs(math.sin(car_state.angle)) - math.fabs(car_state.distance_from_center))
        #self.karma = car_state.distance_from_start * self.karma


if __name__ == '__main__':
    feature_transformer = FeatureTransformer()

    if len(sys.argv) == 2:
        print("Loading model from {}".format(sys.argv[1]))
        control = torch.load(sys.argv[1])
    else:
        control = CarControl(feature_transformer.size, [10, 10])
    sys.argv = sys.argv[:1]

    driver = NeuralDriver(feature_transformer, control, 1)

    from pytocl.main import main as pytocl_main

    pytocl_main(driver)

    print(driver.last_state)
