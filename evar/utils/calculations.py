import numpy as np


class RunningMean:
    """Running mean calculator for arbitrary axis configuration.
    Thanks to https://math.stackexchange.com/questions/106700/incremental-averageing
    """

    def __init__(self, axis):
        self.n = 0
        self.axis = axis

    def put(self, x):
        if self.n == 0:
            self.mu = x.mean(self.axis, keepdims=True)
        else:
            self.mu += (x.mean(self.axis, keepdims=True) - self.mu) / self.n
        self.n += 1

    def __call__(self):
        return self.mu

    def __len__(self):
        return self.n


class RunningVariance:
    """Calculate mean/variance of tensors online.
    Thanks to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, axis, mean):
        self.update_mean(mean)
        self.s2 = RunningMean(axis)

    def update_mean(self, mean):
        self.mean = mean

    def put(self, x):
        self.s2.put((x - self.mean) **2)

    def __call__(self):
        return self.s2()

    def std(self):
        return np.sqrt(self())


class RunningStats:
    def __init__(self, axis=None):
        self.axis = axis
        self.mean = self.var = None

    def put(self, x):
        assert type(x)
        if self.mean is None:
            if self.axis is None:
                self.axis = list(range(len(x.shape)))
            self.mean = RunningMean(self.axis)
            self.var = RunningVariance(self.axis, 0)
        self.mean.put(x)
        self.var.update_mean(self.mean())
        self.var.put(x)

    def __call__(self):
        return self.mean(), self.var.std()
