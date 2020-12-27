import numpy as np
import torch

from modules.plotting import plot_performances
from modules.preprocess import load_tensors, phi

from modules.training import TrainingInstance

from pathlib import Path
import os

os.environ['PYTHONHASHSEED'] = '0'
seed = 12345324
np.random.seed(seed)
torch.random.manual_seed(seed)

base_dir = Path('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/naturalcomputingcw2release')
cache_loc = base_dir / 'cache'
data_loc = base_dir / 'data'
fig_loc = base_dir / 'figures'

epochs = 10000
baseline_inertia = 0.1
baseline_a1 = 1.6
baseline_a2 = 2.4
baseline_population_size = 30
baseline_search_range = 1

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors(data_loc / 'two_spirals.dat')

baseline = TrainingInstance(
        x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val, 
        network_structure=[6,8,1],
        inertia=baseline_inertia, a1=baseline_a1, a2=baseline_a2, population_size=baseline_population_size, search_space=baseline_search_range, 
        seed=seed, epochs=epochs,
        cache_loc=cache_loc
        )

optimal_inertia = 0.1
optimal_a1 = 1.3
optimal_a2 = 2.7
optimal_search_space = 1
optimal_population_size = 30 


optimal = TrainingInstance(
        x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val, 
        network_structure=[6,8,1],
        inertia=optimal_inertia, a1=optimal_a1, a2=optimal_a2, population_size=optimal_population_size, search_space=optimal_search_space, 
        seed=seed, epochs=epochs,
        cache_loc=cache_loc
        )

plot_performances(
    training_instances=[baseline, optimal],
    labels=["Baseline", "Optimized"],
    plot_title="Optimal hyperparameters",
    fitness_name="Binary Cross Entropy loss",
    save_location=fig_loc / 'optimized_vs_baseline.pdf'
    )
