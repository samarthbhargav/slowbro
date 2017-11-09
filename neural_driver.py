from pytocl.driver import Driver
from pytocl.car import State, Command

from neural_net import CarControl, FeatureTransformer


class NeuralDriver(Driver):
    def __init__(self, feature_transformer, network):
        super().__init__()

        self.feature_transformer = feature_transformer
        self.network = network
        

    def drive(self, car_state: State) -> Command:
        """
        Produces driving command in response to newly received car state.
        """

        feature_vector = self.feature_transformer.transform(car_state)
        steering, brake, acceleration = self.network(feature_vector).data

        command = Command()
        command.accelerator = acceleration
        command.brake = brake
        command.steering = steering


        # TODO make handling the gear a part of the neural net
        if not command.gear:
            command.gear = car_state.gear or 1
        if car_state.rpm > 8000:
                command.gear = car_state.gear + 1
        elif car_state.rpm < 2500:
            command.gear = car_state.gear - 1
        
        

        if self.data_logger:
            self.data_logger.log(car_state, command)

        return command


if __name__ == '__main__':
    feature_transformer = FeatureTransformer()
    control = CarControl(feature_transformer.size, [10, 10])
    driver = NeuralDriver(feature_transformer, control)

    from pytocl.main import main as pytocl_main

    pytocl_main(driver)