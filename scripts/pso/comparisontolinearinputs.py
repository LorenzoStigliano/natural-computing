from pathlib import Path
import numpy as np
import torch
from torch import nn

from modules.preprocess import load_tensors, phi
from modules.plotting import plot_performances

from modules.training import TrainingInstance

import os

os.environ['PYTHONHASHSEED'] = '1'
seed = 12345324
np.random.seed(seed)
torch.random.manual_seed(seed)

loss = nn.BCEWithLogitsLoss()

epochs = 10000
inertia = 0.1
a1 = 1.3
a2 = 2.7
search_range = 1
population_size = 30


base_dir = Path('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/naturalcomputingcw2release')
cache_dir = base_dir / 'cache'
data_dir = base_dir / 'data'

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors(data_dir / 'two_spirals.dat')

baseline_instance = TrainingInstance(x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val,
        network_structure=[6,8,1], 
        epochs=epochs, inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_space=search_range, seed=seed,
        cache_loc=cache_dir)
linear_inputs_instance = TrainingInstance(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, 
        network_structure=[2,8,1], 
        epochs=epochs, inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_space=search_range, seed=seed,
        cache_loc=cache_dir)

plot_performances(
    training_instances=[baseline_instance, linear_inputs_instance],
    labels=["Optimized", "Linear inputs"],
    plot_title="Comparison with linear input only network",
    fitness_name="Binary Cross Entropy loss",
    save_location="/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/naturalcomputingcw2release/figures/comparisontolinearinputs.pdf")
