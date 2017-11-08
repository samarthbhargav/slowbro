#! /usr/bin/env python3

import subprocess
import os
import time
import glob
import pickle
from pytocl.main import main
from my_driver import MyDriver

command = "torcs -r /home/student/Documents/torcs-server/quickrace.xml"
subprocess.Popen(command.split())
time.sleep(3)
print("Driver is in slowbro")
main(MyDriver())
os.wait()
list_of_files = glob.glob('drivelogs/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
for x in pickle.load(open(latest_file, "rb")):
    print(x)

