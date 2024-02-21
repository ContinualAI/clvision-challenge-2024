import sys
import time

from avalanche.logging import TextLogger, BaseLogger
from avalanche.logging.text_logging import UNSUPPORTED_TYPES


class ShortTextLogger(TextLogger):

    def __init__(self, file=sys.stdout, truncate_name: bool = True):
        super().__init__(file)
        self.truncate_name = truncate_name
        self._epoch_time = time.time()
    
    def before_training_epoch(self, strategy, *args, **kwargs):
        super().before_training_epoch(strategy, *args, **kwargs)
        self._epoch_time = time.time()
    
    def after_training_epoch(self, strategy, metric_values, **kwargs):
        super(BaseLogger, self).after_training_epoch(strategy, metric_values, **kwargs)
        print(f"Epoch {strategy.clock.train_exp_epochs:4d} | Time: {time.time() - self._epoch_time:6.1f} ", file=self.file, end="")
        self.print_current_metrics()
        self.metric_vals = {}

    def print_current_metrics(self):
        sorted_vals = sorted(self.metric_vals.values(), key=lambda x: x[0])
        for name, x, val in sorted_vals:
            if isinstance(val, UNSUPPORTED_TYPES):
                continue
            val = self._val_to_str(val)
            print_name = name if not self.truncate_name else name.split("/")[0]
            print(f"| {print_name} = {val} ", file=self.file, end="")
        print()

