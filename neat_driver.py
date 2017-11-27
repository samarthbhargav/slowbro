import glob
import os
import pickle
import subprocess
import threading

import neat

from neural_net import FeatureTransformer
from pytocl.car import State, Command
from pytocl.driver import Driver
from pytocl.main import main as pytocl_main
import time

FNULL = open("./logs.txt", 'w')


class BooleanFlag:
    def __init__(self, value):
        self._value = value
        self.lock = threading.Lock()

    def set(self, value):
        with self.lock:
            self._value = value

    @property
    def value(self):
        return self._value


class DriverEnvironment:
    def __init__(self, race_config_path, logs_path="drivelogs/*"):
        self.command = "torcs -r {}".format(os.path.abspath(race_config_path))
        self.logs_path = logs_path

    def get_fitness(self, feature_transformer, network):
        try:
            proc = subprocess.Popen(self.command.split(), stdout=FNULL)

            return_code = proc.poll()

            if return_code is not None:
                raise ValueError("Some error occurred. Either torcs isn't installed or the config file is not present")
            stop_flag = BooleanFlag(False)
            driver = NeatDriver(feature_transformer, network, stop_flag)
            pytocl_main(driver)
            print("ARGH")

            while stop_flag.value is False:
                print("#####", stop_flag.value)
                time.sleep(1)
                proc.kill()

            subprocess.call(["killall", "torcs-bin"])

            list_of_files = glob.glob(self.logs_path)
            latest_file = max(list_of_files, key=os.path.getctime)
            print(latest_file)
            end_state, end_command = pickle.load(open(latest_file, "rb"))

            # better fitness!
            print(end_state)
            print()

            # TODO come up with a better fitness function
            return driver.last_state.distance_raced
        finally:
            if proc:
                try:
                    proc.kill()
                except:
                    # ignore errors
                    ...


class NeatDriver(Driver):
    def __init__(self, feature_transformer, network, shared_flag):
        super().__init__()

        self.feature_transformer = feature_transformer
        self.network = network
        self.call_number = 0
        self.stop_flag = shared_flag
        self.last_state = None

    def drive(self, car_state: State) -> Command:
        """
        Produces driving command in response to newly received car state.
        """
        self.last_state = car_state

        feature_vector = self.feature_transformer.transform(car_state)
        steering, brake, acceleration = self.network.activate(feature_vector)

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

        # TODO early stopping

        if self.call_number % 100 == 0:
            print(command)
            print(car_state.distance_raced)
            print(self.stop_flag.value)
            print()

        if self.call_number > 1000:
            self.stop_flag.set(True)

        self.call_number += 1

        return command

    def on_shutdown(self):
        super().on_shutdown()
        self.stop_flag.set(True)


class Evaluator:
    def __init__(self, race_config_path, feature_transformer):
        self.race_config_path = race_config_path
        self.feature_transformer = feature_transformer

    def evaluate_fitness(self, genomes, config):
        for genome_id, genome in genomes:
            driver_env = DriverEnvironment(self.race_config_path)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = driver_env.get_fitness(self.feature_transformer, net)


if __name__ == '__main__':
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         './config-feedfoward')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))

    evaluator = Evaluator("./quickrace.xml", feature_transformer=FeatureTransformer())

    # Run until a solution is found.
    winner = p.run(evaluator.evaluate_fitness)

    # TODO do the models thing later
    with open("./winner_genome", "wb") as writer:
        pickle.dump(winner, writer)
