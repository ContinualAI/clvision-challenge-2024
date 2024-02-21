from typing import Union, Iterable, Optional, Sequence, Callable

import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.core import BasePlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.base_sgd import TDatasetExperience
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer


class CompetitionTemplate(SupervisedTemplate):

    def __init__(self, model: Module, optimizer: Optimizer, criterion=CrossEntropyLoss(), train_mb_size: int = 1,
                 train_epochs: int = 1, eval_mb_size: Optional[int] = 1, device: Union[str, torch.device] = "cpu",
                 plugins: Optional[Sequence[BasePlugin]] = None,
                 evaluator: Union[EvaluationPlugin, Callable[[], EvaluationPlugin]] = default_evaluator,
                 eval_every=-1, peval_mode="epoch",):
        super().__init__(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device, plugins,
                         evaluator, eval_every, peval_mode)

        self.unlabelled_ds: Optional[AvalancheDataset] = None

    def train(self, experiences: Union[TDatasetExperience, Iterable[TDatasetExperience]],
              unlabelled_ds: Union[AvalancheDataset] = None,
              eval_streams: Optional[Sequence[Union[TDatasetExperience, Iterable[TDatasetExperience]]]] = None, **kwargs):
        self.unlabelled_ds = unlabelled_ds
        super().train(experiences, eval_streams, **kwargs)
