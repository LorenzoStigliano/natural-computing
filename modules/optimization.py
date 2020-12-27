import numpy as np

import torch
from torch.optim import Optimizer


class PSO(Optimizer):
    """
    Pytorch implementation of PSO algorithm. Can be given any compatible model and loss function,
    along with the classical PSO parameters, and can optimise said model's weights using said loss function.

    """

    def __init__(self, features, labels, model, loss, inertia, a1, a2, population_size, search_range, dim, seed, cuda=False):
        self.features = features
        self.labels = labels
        self.model = model
        self.loss = loss

        self.inertia = inertia
        self.a1 = a1
        self.a2 = a2
        self.population_size = population_size
        self.search_range = search_range

        self.dim = dim
        self.seed = seed
        self.cuda = cuda

        self.positions = np.random.uniform(low=-search_range, high=search_range, size=(self.population_size, self.dim))
        self.velocities = np.random.uniform(low=-0.1, high=0.1, size=(self.population_size, self.dim))

        self.best_swarm_position = np.random.uniform(low=-search_range, high=search_range, size=self.dim)
        self.best_swarm_fitness = 1e30

        self.best_particle_positions = np.copy(self.positions)
        self.best_particle_fitnesses = np.array([1e30] * self.population_size)
        

    def step(self, closure=None):
        """
        Executed at each epoch of the PSO algorithm. This is similar to the provided PSO, but has been optimised
        with numpy to ensure potential matrix multiplications are being taken advantage of.
        """
        a1r1 = np.multiply(self.a1, np.random.uniform(low=0, high=1, size=(self.population_size, self.dim)))
        a2r2 = np.multiply(self.a2, np.random.uniform(low=0, high=1, size=(self.population_size, self.dim)))

        best_particle_dif = np.subtract(self.best_particle_positions, self.positions)
        best_swarm_dif = np.subtract(self.best_swarm_position, self.positions)

        self.positions += self.inertia \
                          * self.velocities \
                          + np.multiply(a1r1, best_particle_dif) \
                          + np.multiply(a2r2, best_swarm_dif)

        # if np.any(self.positions.T @ self.positions > 1.0e+18):
        #      raise SystemExit('Most likely divergent: Decrease parameter values')

        with torch.no_grad():
            for i in range(self.population_size):
                self.update_model_weights(self.positions[i])

                if self.cuda:
                    particle_predictions = self.model(self.features.cuda())
                    particle_fitness = self.loss(particle_predictions, self.labels.type(torch.FloatTensor).cuda())
                else:
                    particle_predictions = self.model(self.features)
                    particle_fitness = self.loss(particle_predictions, self.labels.type(torch.FloatTensor))

                if particle_fitness < self.best_particle_fitnesses[i]:
                    self.best_particle_fitnesses[i] = particle_fitness
                    self.best_particle_positions[i] = np.copy(self.positions[i])
                if particle_fitness < self.best_swarm_fitness:
                    self.best_swarm_fitness = particle_fitness
                    self.best_swarm_position = np.copy(self.positions[i])

            self.update_model_weights(self.best_swarm_position)

    def update_model_weights(self, new_position):
        """
        Iterates over the weights of the pytorch model, and updates them with the new best set of weights.
        """
        old_layer_len = 0
        for name, param in self.model.named_parameters():
            if not any(s in name for s in ["weight", "bias"]):
                continue

            new_layer_len = np.prod([i for i in param.size()])
            new_position_layer_1 = new_position[old_layer_len:new_layer_len+old_layer_len]
            old_layer_len = new_layer_len

            param.copy_(torch.tensor(new_position_layer_1).reshape(param.size()))

    def __hash__(self):
        return hash((self.inertia, self.a1, self.a2, self.population_size, self.search_range, self.seed))
