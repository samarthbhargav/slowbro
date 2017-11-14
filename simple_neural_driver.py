from pytocl.driver import Driver
from pytocl.car import State, Command

from train_simple_neural_driver import SimpleCarControl, SimpleFeatureTransformer, SimpleState

import sys
import torch

class SimpleNeuralDriver(Driver):
    def __init__(self, feature_transformer, network):
        super().__init__()

        self.feature_transformer = feature_transformer
        self.network = network
        self.frame = 0
        

    def drive(self, car_state: State) -> Command:
        """
        Produces driving command in response to newly received car state.
        """
        #print(car_state)
        #print()
        feature_vector = self.feature_transformer.transform(car_state)
        steering, brake, acceleration = self.network(feature_vector).data

        command = Command()
        command.accelerator = acceleration
        command.brake = 1 if brake > 0.5 else 0 
        command.steering =  steering


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
            print(command)
            print(car_state.distance_raced)
            print()
        self.frame = (1 + self.frame) % 100000
        return command


if __name__ == '__main__':
    feature_transformer = SimpleFeatureTransformer()

    if len(sys.argv) == 2:
        control = torch.load(sys.argv[1])
    else:
        control = SimpleCarControl([10, 10])
    sys.argv = sys.argv[:1]

    driver = SimpleNeuralDriver(feature_transformer, control)

    from pytocl.main import main as pytocl_main

    pytocl_main(driver)