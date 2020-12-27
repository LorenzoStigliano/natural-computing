#%%

import itertools
import json
import os
import numpy as np
import torch
import csv
from pathlib import Path
from progress.bar import Bar
from modules.ga import GA

from modules.preprocess import load_tensors, phi

base_dir = Path('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/naturalcomputingcw2release')
cache_dir = base_dir / 'cache'
data_dir = base_dir / 'data'

os.environ['PYTHONHASHSEED'] = '0'
seed = 12345324
np.random.seed(seed)
torch.random.manual_seed(seed)

### PSO Hyperparameters
pso_epochs = 1000
inertia = 0.7
a1 = 1.5
a2 = 1.8
population_size = 30
search_range = 10

### GA Hyperparameters
max_hidden_units = 8

D_N_range = [[6,60],[5,50],[4,40],[3,30]]
T_range = [150]
p_m_range = [0.05]
p_c_range = [0.7]

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors(data_dir / 'two_spirals.dat')

param_combinations = list(itertools.product(D_N_range, T_range, p_m_range, p_c_range))
bar = Bar('Hyperparameter set', max=len(param_combinations))

with open(data_dir / "hyperparamsearch/GAalldataFORTandD.csv", 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    data = ['D',  'N', 'T',  'p_m', 'p_c', 'validation_loss', 'validation_accuracy', 'best_network_structure']
    csv_writer.writerow(data)

    for params in param_combinations:
        D = params[0][0]
        N = params[0][1]
        T = params[1]
        p_m = params[2]
        p_c = params[3]
        GA_ = GA(x_train=x_train, y_train=y_train,
                 x_val=x_val, y_val=y_val,
                 D=D, N=N,
                 T=T, p_c=p_c, p_m=p_m, seed=seed, max_hidden_units = max_hidden_units,
                 inertia=inertia, a1=a1, a2=a2, population_size=population_size,
                 search_range=search_range, phi=phi,
                 cache_loc=cache_dir)

        best_network_structure, validation_loss, validation_accuracy = GA_.run(pso_epochs)
        
        best_validation_loss = 100000
        print(f" Validation Loss={validation_loss}")
        if validation_loss < best_validation_loss:
            best_D = D
            best_N = N
            best_T = T
            best_p_m = p_m
            best_p_c = p_c
            best_validation_loss = validation_loss

        bar.next()
        
        data =  [D,  N, T,  p_m, p_c, validation_loss, validation_accuracy, str(best_network_structure)]

        csv_writer.writerow(data)
        
with open(data_dir / "hyperparamsearch/bestparamsGAfindingoptimalTandD.json", 'w') as fp:
    json.dump({"D": best_D, "N": best_N, "T": best_T, "p_m": best_p_m, "p_c": best_p_c, "validation_loss": best_validation_loss}, fp)

