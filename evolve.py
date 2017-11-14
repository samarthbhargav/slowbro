import random

import os
import sys
import time
import glob
import pickle
import subprocess
import copy

import torch
from torch.autograd import Variable

from neural_net import CarControl, FeatureTransformer
from pytocl.main import main as pytocl_main
from neural_driver import NeuralDriver


FNULL = open(os.devnull, 'w')


class EvolutionConfig:
	def __init__(self, n_candidates = 10, 
		n_trails = 100, 
		n_discard = 8, 
		save_frequency=1):
		self.n_candidates = n_candidates # the number of candidates to generate 
		self.n_trails = n_trails # number of trails
		self.n_discard = n_discard # number of candidates to discard
		self.save_frequency = save_frequency


class PyTorchRandomGenerator:
	# TODO Anneal the randomness?
	
	def clone_model(self, model):
		return copy.deepcopy(model)

	def mutate(self, model):
		# return a single mutated model
		new_model = self.clone_model(model)
		#print(vars(new_model))
		for param_name, param in new_model._parameters.items():
			# generate add random noise in [-1, 1]
			param.add_(Variable(torch.rand(param.size()) * 2 - 1))
		return new_model

	def evolve(self, seed_model, n_candidates):
		for i in range(n_candidates):
			yield self.mutate(seed_model)



class SimulatedEvolution:

	def __init__(self, seed_model, feature_transformer, generator, 
					fitness_function, model_save_routine, evolution_config=None):
		self.seed_model = seed_model 
		self.feature_transformer = feature_transformer
		# default params if not defined
		self.evolution_config = evolution_config if evolution_config else EvolutionConfig()

		self.generator = generator
		self.fitness_function = fitness_function
		self.model_save_routine = model_save_routine

	def _log(self, statement):
		print(statement)

	def simulate(self):
		generations = [self.seed_model]
		max_fitness = 0.0
		for trail_number in range(self.evolution_config.n_trails):
			self._log("Executing trail {}".format(trail_number))
			candidate_set = []
			for generation in generations:
				for candidate in self.generator.evolve(generation, self.evolution_config.n_candidates):
					candidate_set.append((candidate, 
						self.fitness_function.evaluate_fitness(self.feature_transformer,  candidate)))

			candidate_set.sort(key=lambda _: _[1])
			# keep the ones that score the highest
			generations = candidate_set[self.evolution_config.n_discard:]
			max_fitness = max(generations, key=lambda _: _[1])
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

class FitnessFunction:
	
	def evaluate_fitness(self, model):
		pass

class DriverEnvironment(FitnessFunction):

	def __init__(self, race_config_path, logs_path = "drivelogs/*"):
		self.command = "torcs -r {}".format(os.path.abspath(race_config_path))
		self.logs_path = logs_path

	def evaluate_fitness(self, feature_transformer, model):
		try: 
			proc = subprocess.Popen(self.command.split(), stdout=FNULL)

			return_code = proc.poll()
			if return_code is not None:
				raise ValueError("Some error occurred. Either torcs isn't installed or the config file is not present")
			
			
			pytocl_main(NeuralDriver(feature_transformer, model))
			os.wait()
			list_of_files = glob.glob(self.logs_path)
			latest_file = max(list_of_files, key=os.path.getctime)
			print(latest_file)
			end_state, end_command = pickle.load(open(latest_file, "rb"))

			# better fitness!
			print(end_state)
			print()

			return end_state.distance_raced
		finally:
			if proc:
				try: 
					proc.kill()
				except:
					# ignore errors
					...


if __name__ == '__main__':
	feature_transformer = FeatureTransformer()
	seed_model = CarControl(feature_transformer.size, [50, 50])
	evolution_config = EvolutionConfig(n_candidates=3, n_discard=1, n_trails=1000)
	evolver = SimulatedEvolution(seed_model, feature_transformer,
					generator=PyTorchRandomGenerator(), 
					fitness_function=DriverEnvironment("./quickrace.xml"),
					model_save_routine=PyTorchSaveModelRoutine("models", "initial_trails"),
					evolution_config=evolution_config)
	evolver.simulate()