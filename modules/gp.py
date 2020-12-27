import numpy as np
import matplotlib.pyplot as plt

import torch

import torch.nn as nn

from modules.training import TrainingInstance

class GP:
    """
    Implementation of genetic programming, tasked with optimising the number of layers and activation types for each
    hidden unit of a neural network. Uses Particle swarm optimisation as its optimisation strategy,
    and Binary cross entropy loss as its loss function.
    """
    
    def __init__(self, x_train, y_train, D, N, T, p_c, p_m, seed, cache_loc,
            max_hidden_units=10, dev="cpu", inertia=0.7, a1=1.5, a2=1.8,
            population_size=30, search_range=1, x_val=None, y_val=None, phi=lambda x:x
            ):

        self.x_train = phi(x_train)
        self.y_train = y_train
        self.x_val = phi(x_val) if x_val is not None else None
        self.y_val = y_val

        self.loss = nn.BCEWithLogitsLoss().to(dev)
        self.dev = dev

        self.D = D  # Dimension of the search space
        self.N = N  # Size of the population of solutions, usually much larger
        self.T = T  # Number of generations. Often needs to be larger than this.
        self.p_c = p_c  # Crossover probability
        self.p_m = p_m  # Mutation probbability
        self.min_hidden_units = 0
        self.max_hidden_units = max_hidden_units

        self.elitism = 0  # A binary switch for whether elitism is used (1) or not (0)
        self.population = self._initalize_population()

        self.inertia = inertia
        self.a1 = a1
        self.a2 = a2
        self.population_size = population_size
        self.search_range = search_range
        self.seed = seed

        self.cache_loc = cache_loc

    def _initalize_population(self):
        """
        Initalize the original population.
        Note: the first layer can only have features from 1-6 and the output needs to be 1.
        """

        start = np.random.binomial(n=1, p=0.5, size=(self.N, 6))
       
        member_genes = []
        for member_index in range(self.N):
            member_hidden_units = []
            for layer_index in range(self.D - 2):
                number_hidden_units = np.random.randint(self.max_hidden_units)
                activation_function_choice = np.random.choice(['A', 'B', 'C'])
                member_hidden_units.append([number_hidden_units, activation_function_choice])
                

            member_genes.append([start[member_index], member_hidden_units[0]])
            for i in range(1, len(member_hidden_units)):
                member_genes[member_index].append(member_hidden_units[i])
            member_genes[member_index].append(1)
        init_pop = member_genes
        return np.reshape(init_pop, (self.N, self.D))

    def _fitness_func(self, num_epochs):
        """
        Evaluate each of the members of the population, the members of the population are of the form [a,b,1]
        We need to train the NN for this structure and then evaluate the fitness after they have been trained.

        Return: - the population sorted form best to worst fitness (smallest to largest values)
                - the fitness of the population
        """


        fitness_list = np.zeros(self.N)
        accuracy_list = np.zeros(self.N)
        j = 0

        
        for member in self.population:
            print(f"Currently training: {member}.")
            
            network_structure = []
            for i in range(len(member)):
                if i == 0:
                    network_structure.append(np.sum(member[i]))
                elif i < len(member)-1:
                    network_structure.append(member[i][0])
                else:
                    network_structure.append(1)

            nonlinearities = []
            for i in range(len(member)):
                if i != 0 and i != len(member)-1 and member[i][0]!= 0:
                    nonlinearities.append(member[i][1])

            training_instance = TrainingInstance(x_train=get_feature_subset(self.x_train, member[0]),
                                                 y_train=self.y_train, x_val=get_feature_subset(self.x_val, member[0]),
                                                 y_val=self.y_val, network_structure=network_structure,
                                                 nonlinearity_keys=nonlinearities,
                                                 inertia=self.inertia, a1=self.a1,a2=self.a2, 
                                                 population_size=self.population_size, search_space=self.search_range, 
                                                 seed=self.seed, epochs=num_epochs,
                                                 cache_loc=self.cache_loc)

            fitness = training_instance.get_current_performances().loc[('val', "fitness")]
            accuracy = training_instance.get_current_performances().loc[('val', "accuracy")]
            fitness_list[j] = fitness
            accuracy_list[j] = accuracy

            print(f"{member} with fitness {fitness}.")
            fitness_indices = np.argsort(fitness_list)
            sorted_pop = [self.population[p] for p in fitness_indices]
            j += 1

        fitness_list = fitness_list[fitness_indices]
        accuracy_list = accuracy_list[fitness_indices]

        return sorted_pop, 1/fitness_list, accuracy_list

    def _roulette_wheel_selection(self, sorted_pop, fitness_list):

        intermediate_pop = np.zeros((self.N, self.D), dtype=object)
        select_from = np.arange(self.N)
        total_fit = np.sum(fitness_list)

        if total_fit == 0:
            total_fit = 1
            relative_fitness = fitness_list + 1 / self.N
        else:
            relative_fitness = fitness_list / total_fit
        mating_population = np.random.choice(select_from, self.N, p=relative_fitness)

        for member in range(len(mating_population)):
            intermediate_pop[member] = sorted_pop[mating_population[member]]

        return intermediate_pop

    def _get_new_generation(self, intermediate_pop):

        new_pop = np.zeros((self.N, self.D), dtype=object)
        parent_list = np.arange(self.N)
        pairings = np.random.choice(parent_list, (2, int(self.N / 2)), replace=False)
        for x in range(np.int(self.N / 2)):
            parent1 = pairings[0][x]
            parent2 = pairings[1][x]
            new_pop[x], new_pop[(self.N - 1) - x] = self._crossover(intermediate_pop[parent1], intermediate_pop[parent2])
        
        self._mutate(new_pop)
        
        return new_pop

    def _crossover(self, parent1, parent2):

        c_point = np.random.randint(0, self.D - 1)  # Crossover point since last element will always be 1
        child1 = np.zeros(self.D, dtype=object)
        child2 = np.zeros(self.D, dtype=object)
        for chromosome in range(c_point):
            child1[chromosome] = parent1[chromosome]
            child2[chromosome] = parent2[chromosome]
        for chromosome in range(self.D - c_point):
            child1[c_point + chromosome] = parent2[c_point + chromosome]
            child2[c_point + chromosome] = parent1[c_point + chromosome]

        return child1, child2

    def _mutate(self, population):

        for member_index in range(len(population)):
            for chromosome_type_index in range(self.D - 1):
                if np.random.rand() < self.p_m:
                    if chromosome_type_index == 0:
                        population[member_index][chromosome_type_index] = [not x if np.random.rand() < self.p_m else x for x in population[member_index][chromosome_type_index]]
                    else:
                        population[member_index][chromosome_type_index][0] = np.random.randint(self.max_hidden_units)
                        population[member_index][chromosome_type_index][1] = np.random.choice(["A","B","C"])

        return population

    def run(self, num_epochs, ax=None, title=None, savefig=None):
        """
        Function which runs the Genetic programming algorithm and returns the best network structure and activations,
        along with its fitness and accuracy after num_epochs epochs.

        The average fitnesses over each epoch can also be plotted if savefig is provided.
        These plots can be combined by passing an axis manually to this function.
        """

        average_fitnesses = []

        convergence_score = 0

        for t in range(self.T):
            # Evaluate fitness for current generation
            sorted_pop, fitness_list, accuracy_list = self._fitness_func(num_epochs)
            # Calculates average fitness at time t for later plotting.
            average_fitnesses.append(np.mean(1/fitness_list))
            # Selection process
            intermediate_pop = self._roulette_wheel_selection(sorted_pop, fitness_list)
            # Update for the next generation, here is where we crossover and mutation

            new_generation = self._get_new_generation(intermediate_pop)
            if new_generation == self.population:
                convergence_score += 1

            if convergence_score == 5:
                print("Likely converged: Population unchanged for 5 generations!")
                break

            print(t), print(self.population)

        if savefig is not None:
            if ax is None:
                fig, ax = plt.subplots()
            if title is not None:
                ax.set_title(title)
            ax.plot(list(range(self.T)), average_fitnesses)
            plt.savefig(savefig)

        sorted_pop, fitness_list, accuracy_list = self._fitness_func(num_epochs)
        return sorted_pop[0], 1/fitness_list[0], accuracy_list[0], ax

def get_feature_subset(data, phi_bool_map):
    """Returns a subsets of the features given a binary map"""
    subset = data[:, np.nonzero(phi_bool_map)]
    return subset.reshape((subset.shape[0], subset.shape[2]))
