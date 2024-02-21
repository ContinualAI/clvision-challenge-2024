import torch
import time
import warnings

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class GPUMemoryChecker(SupervisedPlugin):
    """
    This plugin checks the maximum amount of GPU memory allocated after each
    experience.
    """
    def __init__(self, max_allowed: int = 5000):
        """
        :param max_allowed: Maximum GPU memory allowed in MB.
        :param device: Device for which memory allocation should be checked.
        """

        super().__init__()
        self.max_allowed = max_allowed
        self.gpu_allocated = 0

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        gpu_allocated = torch.cuda.max_memory_allocated()
        gpu_allocated = gpu_allocated // 1000000
        self.gpu_allocated = gpu_allocated
        print(f"MAX GPU MEMORY ALLOCATED: {gpu_allocated} MB")

        if gpu_allocated > self.max_allowed:
            warnings.warn(f"MAX VRAM USAGE WARNING: Current measured maximum VRAM usage {gpu_allocated} MB is over the maximum allowed {self.max_allowed} MB")


class TimeChecker(SupervisedPlugin):
    """
    This plugin checks the amount of time spent after each experience.
    """
    def __init__(self, max_allowed: int = 5000):
        """
        :param max_allowed: Maximum amount of time allowed in minutes.
        """

        super().__init__()
        self.max_allowed = max_allowed
        self.start = time.time()
        self.time_spent = 0

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        time_spent = time.time() - self.start
        time_spent = time_spent // 60
        self.time_spent = time_spent
        print(f"TIME SPENT: {time_spent} MINUTES")

        if time_spent > self.max_allowed:
            warnings.warn(f"MAX TRAINING TIME WARNING: Current training time {self.time_spent} min exceeds the maximum allowed of {self.max_allowed} min.")


__all__ = ["GPUMemoryChecker", "TimeChecker"]
