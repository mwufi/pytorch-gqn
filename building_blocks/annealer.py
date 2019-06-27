
class Annealer(object):
    def __init__(self, init, delta, steps):
        self.init = init
        self.delta = delta
        self.steps = steps
        self.s = 0
        self.data = self.__repr__()
        self.recent = init

    def __repr__(self):
        return {"init": self.init, "delta": self.delta, "steps": self.steps, "s": self.s}

    def __iter__(self):
        return self

    def __next__(self):
        self.s += 1
        value = max(self.delta + (self.init - self.delta) * (1 - self.s / self.steps), self.delta)
        self.recent = value
        return value
