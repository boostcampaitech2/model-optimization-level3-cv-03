"""Custom loss for long tail problem.

- Author: Junghoon Kim
- Email: placidus36@gmail.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomKLLoss:
    """Custom Criterion."""

    def __init__(self, device,alpha,T, fp16=False):
        self.device = device
        self.fp16 = fp16
        self.alpha = alpha
        self.T = T
        self.criterion = nn.KLDivLoss(reduction='batchmean')


    def __call__(self, logits, labels,teacher_outputs):
        """Call criterion."""
        alpha= self.alpha
        T=self.T
        if labels is not None:
            if teacher_outputs is None: ## for test : normal CE
                kl_loss = F.cross_entropy(logits, labels)
            else:
                kl_loss = self.criterion(F.log_softmax(logits/T, dim=1),
                                    F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                    F.cross_entropy(logits, labels) * (1. - alpha)
        else: ## for pseudo labeling : no labels
            kl_loss= self.criterion(
                F.log_softmax(logits, dim=1),
                F.softmax(teacher_outputs, dim=1)
            )

        return kl_loss


class CustomCriterion:
    """Custom Criterion."""

    def __init__(self, samples_per_cls, device, fp16=False, loss_type="softmax"):
        if not samples_per_cls:
            loss_type = "softmax"
        else:
            self.samples_per_cls = samples_per_cls
            self.frequency_per_cls = samples_per_cls / np.sum(samples_per_cls)
            self.no_of_classes = len(samples_per_cls)
        self.device = device
        self.fp16 = fp16

        if loss_type == "softmax":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == "logit_adjustment_loss":
            tau = 1.0
            self.logit_adj_val = (
                torch.tensor(tau * np.log(self.frequency_per_cls))
                .float()
                .to(self.device)
            )
            self.logit_adj_val = (
                self.logit_adj_val.half() if fp16 else self.logit_adj_val.float()
            )
            self.logit_adj_val = self.logit_adj_val.to(device)
            self.criterion = self.logit_adjustment_loss

    def __call__(self, logits, labels):
        """Call criterion."""
        return self.criterion(logits, labels)

    def logit_adjustment_loss(self, logits, labels):
        """Logit adjustment loss."""
        logits_adjusted = logits + self.logit_adj_val.repeat(labels.shape[0], 1)
        loss = F.cross_entropy(input=logits_adjusted, target=labels)
        return loss
