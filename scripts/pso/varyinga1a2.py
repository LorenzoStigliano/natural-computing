import itertools
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from modules.models import BaselineSpiralClassifier
from modules.preprocess import load_tensors, phi

from modules.training import TrainingInstance

plt.style.use('ggplot')

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

epochs = 3000
inertia = 0.7
population_size = 30
search_range = 500

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors(data_loc / 'two_spirals.dat')

a1_range = np.arange(0, 5.2, 0.2)
a2_range = a1_range[::-1]

a1a2_pairs = []
for pair in itertools.product(a1_range, a2_range):
    if sum(pair) == 4.0:
        a1a2_pairs.append(pair)

training_instances = []
for a1a2 in a1a2_pairs:
    training_instance = TrainingInstance(
        instance_name=r"$\a1 = {:.2f}; a2= {:.2f}$".format(a1a2[0], a1a2[1]), model=baseline_model, loss=loss, epochs=epochs,
        inertia=inertia, a1=a1a2[0], a2=a1a2[1], population_size=population_size, search_range=search_range, seed=seed
    )
    training_instance.fit(x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val)
    training_instances.append(training_instance)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

training_accuracies = [training_instance.get_performances().loc[("val", "accuracy")].iloc[-1] for training_instance in training_instances]
ax.set_title("Varying a1 and a2")

a1_xs = [x[0] for x in a1a2_pairs]
a2_xs = [x[1] for x in a1a2_pairs]
ax.scatter(a1_xs, a2_xs, training_accuracies)
ax.set_xlabel("a1")
ax.set_ylabel("a2")
ax.set_zlabel("Accuracy")
plt.show()
