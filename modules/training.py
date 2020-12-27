import os
from pathlib import Path

import pandas as pd
import torch
from progress.bar import Bar
from torch.nn import BCEWithLogitsLoss

from modules.models import GenericSpiralClassifier, BaselineSpiralClassifier
from modules.optimization import PSO


def _get_accuracy(y_train, y_train_preds):
    class_classified = (y_train_preds > 0.5).float()
    accuracy = sum(y_train[i] == class_classified[i] for i in range(len(class_classified))) / y_train_preds.shape[0]
    return accuracy


class TrainingInstance:
    """
    Convenience class for training and passing around the results of a Particle Swarm Optimisation.
    This class handles caching of results implicitly, such that if we try to train the same network twice,
    it will instead load the results from disk.

    x_train: X training data
    y_train: y training data

    !!! PSO params !!!
    inertia: float
    a1: float
    a2: float
    population_size: int
    search_space: int
    epochs: int

    seed: Seed for all random number generation.

    cache_loc: Path to local cache

    network_structure: List[int] -> A list the length of the desired network,
        with entries corresponding to the number of hidden units at that layer.
    nonlinearity_keys: List[char] -> List of characters corresponding to a nonlinearity function for each layer.

    x_val: X validation data
    y_val: y validation data

    device: whether to run on cpu or some other device e.g. a CUDA gpu device.

    verbose: Toggle logging verbosity

    """

    def __init__(self, x_train, y_train, inertia, a1, a2, population_size, search_space, epochs, seed, cache_loc, network_structure=None, nonlinearity_keys=None, x_val=None, y_val=None, device="cpu", verbose=False):

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.pso_params = {"inertia": inertia, "a1": a1, "a2": a2, "search_space": search_space, "population_size": population_size}

        if network_structure is not None:
            self.network_structure = list(filter(lambda a: a != 0, network_structure))
        else:
            self.network_structure = [6,8,1]

        self.nonlinearity_keys = ["A" for _ in range(len(self.network_structure) - 2)] if nonlinearity_keys is None else nonlinearity_keys


        self.model = GenericSpiralClassifier(self.network_structure, self.nonlinearity_keys).to(device) if network_structure is not None else BaselineSpiralClassifier().to(device)
        self.loss = BCEWithLogitsLoss().to(device)
        self.optimizer = PSO(x_train, y_train, model=self.model, loss=self.loss, inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_range=search_space, seed=seed, dim=self.get_n_trainable_params())
        self.epochs = epochs
        self.device = device

        self.performance_cache = cache_loc / f"performances/{hash(self)}"
        self.model_cache = cache_loc / f"models/{hash(self)}"

        self.verbose = verbose

        if os.path.exists(self.model_cache):
            self.performances = self._load_cached_performances()
            self._load_cached_model_state()
            print("Loaded from disk.")
        else:
            self.performances = pd.DataFrame(index=pd.MultiIndex.from_product([["train", "val"], ["fitness", "accuracy"]]),
                                              columns=[i for i in range(self.epochs)]
                                              )
            self.progress_bar = Bar('PSO epoch', max=self.epochs)
            self._fit()

    def _fit(self):
        for i in range(self.epochs):
            y_train_preds = self.model(self.x_train)
            fitness = self.loss(y_train_preds, self.y_train)
            accuracy = _get_accuracy(self.y_train, y_train_preds)

            self.performances.loc[("train", "fitness")][i] = fitness
            self.performances.loc[("train", "accuracy")][i] = accuracy

            if self.x_val is not None and self.y_val is not None:
                y_val_preds = self.model(self.x_val)
                val_fitness = self.loss(y_val_preds, self.y_val)
                val_acc = _get_accuracy(self.y_val, y_val_preds)
                self.performances.loc[("val", "fitness")][i] = val_fitness
                self.performances.loc[("val", "accuracy")][i] = val_acc

            self.optimizer.step()

            if self.verbose:
                print(f"Epoch {i}: Fitness = {fitness}; Training Acc = {accuracy}")

            self.progress_bar.next()

        self._cache_performances()
        self._cache_model_state()


    def get_n_trainable_params(self):
        n = 0
        for name, param in self.model.named_parameters():
            if any(x in name for x in ["weight", "bias"]):
                n += param.numel()
        return n

    def get_current_performances(self):
        return self.performances.iloc[:, -1:]

    def _load_cached_performances(self):
        return pd.read_hdf(self.performance_cache, key="accuracies")
    
    def _cache_performances(self):
        self.performances.to_hdf(self.performance_cache, key="accuracies")

    def _load_cached_model_state(self):
        self.model.load_state_dict(torch.load(self.model_cache))

    def _cache_model_state(self):
        torch.save(self.model.state_dict(), self.model_cache)

    def _get_model_hash(self):
        return hash("".join(str(self.network_structure)) + "".join(self.nonlinearity_keys))

    def __hash__(self):
        return hash((self.epochs, self._get_model_hash(), hash(self.optimizer), self.device))


