from pathlib import Path

import numpy as np
import torch
from torch import nn

from modules.models import BaselineSpiralClassifier
from modules.preprocess import load_tensors, phi
from modules.plotting import plot_performances

from modules.training import TrainingInstance

os.environ['PYTHONHASHSEED'] = '0'
seed = 12345324
np.random.seed(seed)
torch.random.manual_seed(seed)

base_dir = Path('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/naturalcomputingcw2release')
cache_loc = base_dir / 'cache'
data_loc = base_dir / 'data'
fig_loc = base_dir / 'figures'

baseline_model = BaselineSpiralClassifier()

loss = nn.BCEWithLogitsLoss()

epochs = 10000
inertia = 0.7
a1 = 1.5
a2 = 1.8
population_size = 30
search_range = 10

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/data/two_spirals.dat')

baseline_instance = TrainingInstance(x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val, epochs=epochs, inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_space=search_range, seed=seed)
initial_hyperparameter_best = TrainingInstance(x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val, epochs=epochs, inertia=0.7, a1=2.8, a2=1.2, population_size=population_size, search_space=1, seed=seed)

plot_performances(
    training_instances=[baseline_instance, initial_hyperparameter_best],
    plot_title="Hyperparameter optimization starting point",
    fitness_name="Binary Cross Entropy loss",
    save_location=fig_loc / "shallowhyperparameterbest.pdf")
