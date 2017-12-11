import sys
import math
import torch

from random import random
from neural_net import CarControl, FeatureTransformer
from simple_neural_driver import *


class NeuralDriver(Driver):
    def __init__(self, feature_transformer, network, life_span):
        super().__init__()

        self.feature_transformer = feature_transformer
        self.network = network
        self.frame = 0
        self.last_state = None
        self.prev_state = None
        self.karma = 1
        self.life_span = life_span * 500
        self.standard = (random() > 0.5)
        self.command = None

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

        '''
        if self.last_state is None:
            self.prev_state = car_state
        else:
            self.prev_state = self.last_state
        '''
        self.last_state = car_state
        self.command = command

        if self.frame % 10 == 0:
            high_speed_reward_criteria = 150
            if car_state.distance_from_start < 500:
                high_speed_reward_criteria = 100
            high_speed_reward = 10 if math.fabs(car_state.speed_x) > high_speed_reward_criteria else -10000
            racing_reward = (car_state.distance_from_start / 10000) * (car_state.speed_x / 300) * 1000
            no_braking_reward = 10 if command.brake == 0 else -10000
            no_damage_reward = 10 if car_state.damage == 0 else -10000
            driver_on_road_reward = 10 if math.fabs(car_state.distance_from_center) < 0.85 else -10000
            self.karma += (racing_reward * (0.5) + no_damage_reward * (1.5) + no_braking_reward * (2) + driver_on_road_reward * (5) + high_speed_reward * (0.5)) * 0.0001

        self.frame = (1 + self.frame) % 100000

        return command

    def calculate_karma(self):
        car_state = self.last_state
        karma = 0
        if math.fabs(car_state.distance_from_center) > 0.99:
            karma -= 500
        else:
            karma += car_state.speed_x * (math.cos(car_state.angle) - math.fabs(math.sin(car_state.angle)) - math.fabs(car_state.distance_from_center))
        return karma

    def calculate_karma_alternate(self):
        car_state = self.last_state
        car_state_1 = self.prev_state
        karma = 0
        if math.fabs(car_state.distance_from_center) > 0.99:
            karma -= 5
        elif math.fabs(car_state.distance_from_center) > 0.85:
            karma += (car_state.distance_from_center - car_state_1.distance_from_center) * (math.cos(car_state.angle) - math.fabs(math.sin(car_state.angle)))
        else:
            karma += (car_state.distance_from_center - car_state_1.distance_from_center) * (math.cos(car_state.angle) - math.fabs(math.sin(car_state.angle)) - (((math.fabs(car_state.distance_from_center) - 0.85) ** 2)/0.0225))
        return karma


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
