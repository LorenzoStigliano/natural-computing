import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from progress.bar import Bar
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
pso_epochs = 1000
inertia = 0.7
a1 = 1.5
a2 = 1.8
population_size = 30
search_range = 10

### GA Hyperparameters
max_hidden_units = 10
D_range = [10]
N_range = [16]
T_range = [1000]
p_m_range = [0.05]
p_c_range = [0.7]


x_train, y_train, x_val, y_val, x_test, y_test = load_tensors(data_loc / 'two_spiral.dat')


param_combinations = list(itertools.product(D_range, N_range, T_range, p_m_range, p_c_range))
bar = Bar('Hyperparameter set', max=len(param_combinations))

best_validation_loss=1e30

for params in param_combinations:
    D = params[0]
    N = params[1]
    T = params[2]
    p_m = params[3]
    p_c = params[4]
    GP_ = GP(x_train=x_train, y_train=y_train,
             x_val=x_val, y_val=y_val,
             D=D, N=N,
             T=T, p_c=p_c, p_m=p_m, seed=seed, max_hidden_units = max_hidden_units,
             inertia=inertia, a1=a1, a2=a2, population_size=population_size,
             search_range=search_range, phi=phi, cache_loc=cache_loc)

    best_network_structure, validation_loss, validation_accuracy = GP_.run(pso_epochs)

    print(f" Validation Loss={validation_loss}")
    if validation_loss < best_validation_loss:
        best_D = D
        best_N = N
        best_T = T
        best_p_m = p_m
        best_p_c = p_c
        best_validation_loss = validation_loss

    bar.next()

with open(data_loc / "hyperparamsearch/bestparamsGP.json", 'w') as fp:
     json.dump({"D": best_D, "N": best_N, "T": best_T, "p_m": best_p_m, "p_c": best_p_c, "validation_loss": best_validation_loss}, fp)

