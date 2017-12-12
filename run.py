#! /usr/bin/env python3

import sys
sys.path.insert(0,'./')

from pytocl.main import main
from my_driver import MyDriver

from neural_driver import NeuralDriver
from neural_net import FeatureTransformer

import torch

if __name__ == '__main__':
    feature_transformer = FeatureTransformer()
    control = torch.load("models/supervisor/3.pty")
    main(NeuralDriver(feature_transformer, control, 100000))
