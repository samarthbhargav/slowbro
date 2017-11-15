#! /usr/bin/env python3


import os
import sys
import time
import glob
import pickle
import subprocess

from pytocl.main import main
from my_driver import MyDriver
from simple_neural_driver import SimpleNeuralDriver

def run(driver, config_path):
	command = "torcs -r {}".format(os.path.abspath(config_path))

	sys.argv = sys.argv[:1]

	try: 
		proc = subprocess.Popen(command.split())
		time.sleep(3)

		return_code = proc.poll()
		if return_code is not None:
			raise ValueError("Some error occurred. Either torcs isn't installed or the config file is not present")

		print("Driver is in slowbro")
		
		main(driver)
		os.wait()
		list_of_files = glob.glob('drivelogs/*') # * means all if need specific format then *.csv
		latest_file = max(list_of_files, key=os.path.getctime)
		print(latest_file)
		
		end_state, end_command = pickle.load(open(latest_file, "rb"))

	finally:
		if proc:
			try: 
				proc.kill()
			except:
				# ignore errors
				...





if __name__ == '__main__':
	assert len(sys.argv) == 2, "Path to config is missing"
	run(SimpleNeuralDriver(), sys.argv[1])

