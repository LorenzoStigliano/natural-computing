from pathlib import Path
import numpy as np
import torch
import os

from modules.gp import GP
from modules.preprocess import load_tensors, phi

os.environ['PYTHONHASHSEED'] = '0'
seed = 12345324
np.random.seed(seed)
torch.random.manual_seed(seed)

base_dir = Path('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/naturalcomputingcw2release')
cache_loc = base_dir / 'cache'
data_loc = base_dir / 'data'
figure_loc = base_dir / 'figures'

### PSO Hyperparameters
pso_epochs = 1
inertia = 0.7
a1 = 1.5
a2 = 1.8
population_size = 30
search_range = 10

### GA Hyperparameters
max_hidden_units = 8
D = 6
N = 10
T = 250
p_m = 0.05
p_c = 0.7

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors(data_loc / "two_spirals.dat")

GP_ = GP(x_train=x_train, y_train=y_train,
         x_val=x_val, y_val=y_val,
         D=D, N=N,
         T=T, p_c=p_c, p_m=p_m, seed=seed, max_hidden_units = max_hidden_units,
         inertia=inertia, a1=a1, a2=a2, population_size=population_size,
         search_range=search_range, phi=phi,
         cache_loc=cache_loc)

best_network_structure, validation_loss, validation_accuracy, _ = GP_.run(pso_epochs, savefig=figure_loc / 'gp_example.pdf', title="GP example")
