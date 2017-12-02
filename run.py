#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver

from neural_driver import NeuralDriver
from neural_net import FeatureTransfomer


if __name__ == '__main__':
    feature_transfomer = FeatureTransfomer()
    control = torch.load("models/supervisor/3.pty")
    main(NeuralDriver(feature_transformer, control))
