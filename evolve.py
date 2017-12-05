import copy
import os
import random
import subprocess

import torch

from neural_driver import NeuralDriver
from neural_net import FeatureTransformer
from pytocl.main import main as pytocl_main

FNULL = open(os.devnull, 'w')


class EvolutionConfig:
    def __init__(self, n_candidates=10,
                 n_trails=100,
                 n_discard=8,
                 save_frequency=1,
                 mutation_normal_mean=0,
                 mutation_normal_std=0.001,
                 mutation_probability=0.3):
        self.n_candidates = n_candidates  # the number of candidates to generate
        self.n_trails = n_trails  # number of trails
        self.n_discard = n_discard  # number of candidates to discard
        self.save_frequency = save_frequency
        self.mutation_normal_mean = mutation_normal_mean
        self.mutation_normal_std = mutation_normal_std
        self.mutation_probability = mutation_probability


def add_normal_noise(tensor, mean, std_dev):
    means = torch.zeros(tensor.size())
    means.fill_(mean)
    noise = torch.normal(means, std_dev)
    tensor.data.add_(noise)


class Mutator:
    def clone_model(self, model):
        return copy.deepcopy(model)

    def mutate(self, model, config: EvolutionConfig):

        # return a single mutated model
        new_model = self.clone_model(model)

        if random.random() >= config.mutation_probability:
            return new_model

        # print("\n".join(str(p) for p in new_model.parameters()))
        for param in new_model.parameters():
            add_normal_noise(param, config.mutation_normal_mean, config.mutation_normal_std)
        # print("#" * 100)
        # print("\n".join(str(p) for p in new_model.parameters()))

        return new_model

    def evolve(self, seed_model, config: EvolutionConfig):
        for i in range(config.n_candidates):
            yield self.mutate(seed_model, config)


class SimulatedEvolution:
    def __init__(self, seed_models, feature_transformer, mutator,
                 fitness_function, model_save_routine, evolution_config=None):
        self.seed_models = seed_models
        self.feature_transformer = feature_transformer

        # default params if not defined
        self.evolution_config = evolution_config if evolution_config else EvolutionConfig()

        self.mutator = mutator
        self.fitness_function = fitness_function
        self.model_save_routine = model_save_routine

    def _log(self, statement):
        print(statement)

    def simulate(self):
        generations = self.seed_models
        for trail_number in range(self.evolution_config.n_trails):
            self._log("Evolving Generation {}".format(trail_number))
            candidate_set = []
            for generation in generations:
                for candidate in self.mutator.evolve(generation, self.evolution_config):
                    candidate_set.append((candidate,
                                          self.fitness_function.evaluate_fitness(self.feature_transformer, candidate, trail_number)))
                    print([c[1] for c in candidate_set])

            candidate_set.sort(key=lambda _: _[1])
            # keep the ones that score the highest
            generations = candidate_set[self.evolution_config.n_discard:]
            max_fitness = max(generations, key=lambda _: _[1])[1]
            generations = [g[0] for g in generations]
            print("Max fitness: {}".format(max_fitness))
            if trail_number % self.evolution_config.save_frequency == 0:
                self.model_save_routine.save(random.choice(generations), trail_number)


class PyTorchSaveModelRoutine:
    def __init__(self, directory, experiment_name):
        self.directory = directory
        self.experiment_name = experiment_name

        self._mkdir(self.directory)

        self.path_ = os.path.join(self.directory, self.experiment_name)
        self._mkdir(self.path_)

    def _mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def save(self, model, n_trail):
        torch.save(model, os.path.join(self.path_, "{0:03d}_model.pty".format(n_trail)))


class DriverEnvironment:
    def __init__(self, race_config_folder, logs_path="drivelogs/*"):
        files = os.listdir(race_config_folder)
        files = [os.path.join(race_config_folder, f) for f in files]
        race_config_path = random.choice(files)
        print("Running file: {}".format(race_config_path))
        self.command = "torcs -r {}".format(os.path.abspath(race_config_path))
        self.logs_path = logs_path

    def evaluate_fitness(self, feature_transformer, model, generation):
        proc = None
        try:
            proc = subprocess.Popen(self.command.split(), stdout=FNULL)

            return_code = proc.poll()
            if return_code is not None:
                raise ValueError("Some error occurred. Either torcs isn't installed or the config file is not present")

            neural_driver = NeuralDriver(feature_transformer, model, generation)
            pytocl_main(neural_driver)
            os.wait()
            end_state = neural_driver.last_state
            fitness = neural_driver.karma

            return fitness
        finally:
            if proc:
                try:
                    proc.kill()
                except:
                    # ignore errors
                    ...


if __name__ == '__main__':
    seed_models = ["models/supervisor/77_intermediate.pty", "models/supervisor/3.pty"]
    seed_models = [torch.load(sm) for sm in seed_models]
    feature_transformer = FeatureTransformer()

    evolution_config = EvolutionConfig(n_candidates=3, n_discard=1, n_trails=1000)
    evolver = SimulatedEvolution(seed_models, feature_transformer,
                                 mutator=Mutator(),
                                 fitness_function=DriverEnvironment("./config_files"),
                                 model_save_routine=PyTorchSaveModelRoutine("models", "3_77"),
                                 evolution_config=evolution_config)
    evolver.simulate()
