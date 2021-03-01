from enum import Enum

__all__ = ("DataType", "Algorithm", "Dataset", "VariationalDistribution")


class DataType(Enum):
    single = 1
    float32 = 2

    double = 11
    float64 = 12

    def to_torch_dtype(self):
        import torch
        if self.value < 10:
            return torch.float32
        else:
            return torch.float64

    def to_numpy_dtype(self):
        import numpy as np
        if self.value < 10:
            return np.float32
        else:
            return np.float64

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return DataType[s]
        except KeyError:
            return s


class Algorithm(Enum):
    FALKON = 'falkon'
    LOGISTIC_FALKON = 'falkon-cls'
    EIGENPRO = 'eigenpro'
    GPYTORCH_REG = 'gpytorch-reg'
    GPFLOW_REG = 'gpflow-reg'
    GPYTORCH_CLS = 'gpytorch-cls'
    GPFLOW_CLS = 'gpflow-cls'
    GPYTORCH_SGPR = 'gpytorch-sgpr'
    GPFLOW_SGPR = 'gpflow-sgpr'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)


class Dataset(Enum):
    TIMIT = 'timit'
    MILLIONSONGS = 'millionsongs'
    HIGGS = 'higgs'
    TAXI = 'taxi'
    YELP = 'yelp'
    FLIGHTS = 'flights'
    FLIGHTS_CLS = 'flights-cls'
    SUSY = 'susy'
    MNIST_SMALL = 'mnist-small'
    SVHN = 'svhn'
    MNIST = 'mnist'
    CIFAR10 = 'cifar10'
    HOHIGGS = 'ho-higgs'
    ICTUS = 'ictus'
    SYNTH01NOISE = 'synth-01noise'
    CHIET = 'chiet'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)


class VariationalDistribution(Enum):
    FULL = 'full'
    DIAG = 'diag'
    DELTA = 'delta'
    NATGRAD = 'natgrad'
    TRIL_NATGRAD = 'tril_natgrad'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
