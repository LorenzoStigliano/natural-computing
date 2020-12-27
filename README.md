This codebase is a useful framework for analysing the effects of PSO, genetic algorithms and genetic programming on the optimisation of neural networks.
Specifically, this base is geared towards analysing the two sprirals dataset provided in the coursework specs, but much functionality can be extended towards more general usage.

Classes which encapsulate the functionality of PSO, genetic algorithms and genetic programming can be found in modules/pso, modules/ga, and modules/gp respectively.
For basic example usage, consult scripts/pso/pso_example.py scripts/ga/ga_example.py, and scripts/gp/gp_example.py respectively.

A critical feature of this framework is the ability to implicitly cache models/performances. This functionality can be found in modules/training

Finally, modules/plotting provides simple functionality to plot the training and validation performance over time given a list of TrainingInstances.