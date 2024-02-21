# Implement your strategy in this file
import copy
from typing import List, Optional, Iterator

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from torch.utils.data import DataLoader

from strategies.competition_template import CompetitionTemplate


class LwFUnlabelled(CompetitionTemplate):
    """
    Implemention of MyStrategy.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion=CrossEntropyLoss(), train_mb_size: int = 1,
                 train_epochs: int = 1, eval_mb_size: Optional[int] = 1, device="cpu",
                 plugins: Optional[List[SupervisedPlugin]] = None, evaluator=default_evaluator(), eval_every=-1,
                 peval_mode="epoch", ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        super().__init__(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device,
                         plugins, evaluator, eval_every, peval_mode, )
        self.unlabelled_alpha = 1.0
        self.old_model: Optional[Module] = None
        self.unlabelled_dl: Optional[DataLoader] = None
        self.unlabelled_iterator: Optional[Iterator] = None

    def _before_training_exp(self, **kwargs):
        # Callback before every task
        # Do not remove super call as plugins might not work properly otherwise
        super()._before_training_exp(**kwargs)
        # Create Dataloader for Knowledge Distillation of unlabelled samples!
        if self.old_model is not None:
            self.unlabelled_dl = DataLoader(self.unlabelled_ds, batch_size=self.train_mb_size, shuffle=True)
            self.unlabelled_iterator = iter(self.unlabelled_dl)

    def training_epoch(self, **kwargs):
        # You can implement your custom training loop here
        # Defaults to a base training loop (see SGDUpdate class)
        # Be careful to add all necessary plugin calls as Avalanche Plugins could not work correctly otherwise!
        super().training_epoch(**kwargs)

    def forward(self):
        # If you need to adjust the forward pass for your model during training
        # you can adjust it here
        return super().forward()

    def criterion(self):
        # Implement your own loss criterion here if needed
        # By default self._criterion gets called which defaults to the CrossEntropy loss
        return super().criterion()

    def _before_backward(self, **kwargs):
        # triggers before backpropagation of the calculated loss from criterion.
        # You can add additional loss terms here. (modify self.loss += ...)
        # For example weight regularization or knowledge distillation
        super()._before_backward(**kwargs)

        if self.old_model is not None and self.unlabelled_iterator is not None:
            try:
                batch_unlabelled = next(self.unlabelled_iterator)
            except StopIteration:
                try:
                    self.unlabelled_iterator = iter(self.unlabelled_dl)
                    batch_unlabelled = next(self.unlabelled_iterator)
                except Exception:
                    batch_unlabelled = None
                    self.unlabelled_iterator = None

            if batch_unlabelled is not None:
                batch_unlabelled = batch_unlabelled.to(self.device)
                pred_old_model = self.old_model(batch_unlabelled)
                old_size = pred_old_model.size(1)
                pred_current_model = self.model(batch_unlabelled)[:, :old_size]

                def _distillation_loss(out, target, temperature):
                    log_p = torch.log_softmax(out / temperature, dim=1)
                    q = torch.softmax(target / temperature, dim=1)
                    return torch.nn.functional.kl_div(log_p, q, reduction="batchmean")

                unlabelled_loss = self.unlabelled_alpha * _distillation_loss(pred_current_model, pred_old_model, 2.0)
                self.loss += unlabelled_loss

    def _after_training_exp(self, **kwargs):
        # Callback after every training task
        # Do not remove super call as plugins might not work properly otherwise
        super()._after_training_exp(**kwargs)

        # Copy Old Model and Freeze
        self.old_model = copy.deepcopy(self.model)
        self.old_model.requires_grad_(False)  # Freeze Model
        self.old_model.eval()
