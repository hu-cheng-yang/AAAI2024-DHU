from typing import Callable

import random
import numpy as np
import pandas as pd
import copy
from collections import defaultdict

import torch
import torch.utils.data
import torchvision
from torch.utils.data.sampler import Sampler



class banlanceDatasetClassSampler(Sampler):
    def __init__(self, labels, batchsize):
        self.indices = []
        self.labels = int(np.max(labels) + 1)
        assert batchsize % self.labels == 0
        self.dataperclass = batchsize // self.labels
        self.indexpool = []
        for i in range(self.labels):
            self.indexpool.append([])
        for index, label in enumerate(labels):
            self.indexpool[label].append(index)
        minLength = min([len(pool) for pool in self.indexpool]) // self.dataperclass
        for index in range(minLength):
            subsequence = []
            for i in range(self.labels):
                for j in range(self.dataperclass):
                    rindex = random.randint(0, len(self.indexpool[i]) - 1)
                    subsequence.append(self.indexpool[i][rindex])
                    del self.indexpool[i][rindex]
            assert len(subsequence) == batchsize
            random.shuffle(subsequence)
            self.indices.extend(subsequence)
    
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class banlanceDatasetClassSampler1(Sampler):
    def __init__(self, labels, batchsize):
        self.indices = []
        self.labels = int(np.max(labels) + 1)
        refined_labels = [i for i in range(0, self.labels, 2)] + [i for i in range(1, self.labels, 2)]
        # print(labels)
        assert batchsize % (self.labels + 2) == 0
        smallbatch = batchsize // (self.labels + 2)
        self.dataperclass = [1 * smallbatch, 1 * smallbatch, 1 * smallbatch, 1 * smallbatch, 2 * smallbatch, 2 * smallbatch] # batchsize // self.labels
        # print(self.dataperclass)
        self.indexpool = []
        for i in range(self.labels):
            self.indexpool.append([])
        for index, label in enumerate(labels):
            self.indexpool[label].append(index)
        minLength = min([ (len(pool) // self.dataperclass[index]) for index, pool in enumerate(self.indexpool)])
        for index in range(minLength):
            subsequence = []
            for i in refined_labels:
                for j in range(self.dataperclass[i]):
                    rindex = random.randint(0, len(self.indexpool[i]) - 1)
                    subsequence.append(self.indexpool[i][rindex])
                    del self.indexpool[i][rindex]
            assert len(subsequence) == batchsize, "Length of subsequence: {}, batchsize: {}".format(len(subsequence), batchsize)
            random.shuffle(subsequence)
            self.indices.extend(subsequence)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)




class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, dataset, labels=None, indices=None, class_choice="balance", callback_get_label=None):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.indices = range(len(dataset)) if indices is None else indices
        self.callback_get_label = callback_get_label

        self.labels = self._get_labels(dataset) if labels is None else labels
        self.num_classes = np.max(self.labels) + 1
        self.num_label_classes = np.max(self.labels) + 1
        self.labels = np.eye(self.num_classes)[self.labels]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(0, self.num_label_classes):
            lst = np.where(self.labels[class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled" or self.class_choice == "debug":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        else:
            class_ = None
        return class_

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.indices)
