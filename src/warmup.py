class QuadraticWarmup:
    def __init__(self, warmup_steps, base_lr):
        self.warmup_steps = max(1, int(warmup_steps))
        self.base_lr = float(base_lr)

    def factor(self, step):
        r = float(step) / float(self.warmup_steps)
        if r >= 1.0:
            return 1.0
        return r * r

    def lr(self, step):
        return self.base_lr * self.factor(step)
