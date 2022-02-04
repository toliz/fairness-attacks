import torch
from torch import nn

from torch import nn, Tensor
from torchmetrics import Metric


class SPD(Metric):
    def __init__(self, dist_sync_on_step=False, use_abs=True):
        """
        A torch metric calculating the statistical parity based on the formula:
        SPD = abs(p(predicted=+1|x in advantaged) - p(predicted=+1|x in disadvantaged))

        Args:
            dist_sync_on_step: Synchronize metric state across processes at each forward() before returning the value at
                the step.
            use_abs: use the absolute value of the SPD
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("preds_adv_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("preds_dis_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_adv", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_dis", default=torch.tensor(0), dist_reduce_fx="sum")

        self.use_abs = use_abs

    def update(self, preds: torch.Tensor, adv_mask: torch.BoolTensor):
        """
        Update the states needed to calculate the SPD.

        Args:
            preds: predictions of model
            adv_mask: advantage mask
        """

        # Advantaged and disadvantaged predictions
        preds_adv = preds[adv_mask]
        preds_dis = preds[~adv_mask]

        # Update the number of advantaged and disadvantaged points predicted as positive
        self.preds_adv_pos += len(preds_adv[preds_adv == 1])
        self.preds_dis_pos += len(preds_dis[preds_dis == 1])

        # Update the number of advantaged and disadvantaged points
        self.num_adv += len(preds_adv)
        self.num_dis += len(preds_dis)

    def compute(self):
        """
        Computes the SPD based on the current states

        Returns: the SPD
        """

        # Probability that the model predicts as positive a data point that has advantage
        p_adv = self.preds_adv_pos / max(self.num_adv, 1)

        # Probability that the model predicts as positive a data point that has disadvantage
        p_dis = self.preds_dis_pos / max(self.num_dis, 1)

        # Calculate SPD
        spd = p_adv - p_dis
        if self.use_abs:
            spd = abs(spd)

        return spd.item()


class EOD(Metric):
    """

    """
    def __init__(self, dist_sync_on_step=False, use_abs=True):
        """
        A torch metric calculating the equal opportunity difference based on the formula:
        EOD = abs(p(predicted=+1|x in advantaged, ground_truth=+1) - p(predicted=+1|x in disadvantaged, ground_truth=+1))

        Args:
            dist_sync_on_step: Synchronize metric state across processes at each forward() before returning the value at
                the step.
            use_abs: use the absolute value of the EOD
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("preds_adv_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("preds_dis_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_adv", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_dis", default=torch.tensor(0), dist_reduce_fx="sum")

        self.use_abs = use_abs

    def update(self, preds: torch.Tensor, targets: torch.Tensor, adv_mask: torch.BoolTensor):
        """
        Update the states needed to calculate the EOD.

        Args:
            preds: predictions of model
            targets: ground truth labels
            adv_mask: advantage mask
        """

        # Advantaged and disadvantaged predictions with label +1
        preds_adv = preds[torch.logical_and(adv_mask, targets.bool())]
        preds_dis = preds[torch.logical_and(~adv_mask, targets.bool())]

        # Update the number of advantaged and disadvantaged points with label +1 predicted as positive
        self.preds_adv_pos += len(preds_adv[preds_adv == 1])
        self.preds_dis_pos += len(preds_dis[preds_dis == 1])

        # Update the number of advantaged and disadvantaged points with target +1
        self.num_adv += torch.logical_and(adv_mask, (targets.bool())).sum()
        self.num_dis += torch.logical_and(~adv_mask, (targets.bool())).sum()

    def compute(self):
        """
        Compute the EOD based on the current states.

        Returns:the EOD
        """

        # Probability that the model predicts as positive a data point that has advantage and label +1
        p_adv = self.preds_adv_pos / max(self.num_adv, 1)

        # Probability that the model predicts as positive a data point that has disadvantage and label +1
        p_dis = self.preds_dis_pos / max(self.num_dis, 1)

        # Calculate the EOD
        eod = p_adv - p_dis
        if self.use_abs:
            eod = abs(eod)

        return eod.item()


class FairnessLoss(nn.Module):
    """The Decision Boundary Covariance loss as defined by Zafar et. al
    (https://arxiv.org/abs/1507.05259).
    
    Currently, this loss supports only binary classification problems solved
    by linear models.
    """
    
    def __init__(self, sensitive_attribute_idx: int):
        super().__init__()
        
        self.sensitive_attribute_idx = sensitive_attribute_idx
        
    def forward(self, X: Tensor, W: Tensor, b: Tensor):
        # Assert we have linear model for binary classification
        assert W.ndim == 2 and b.shape == torch.Size([1])
        
        # Get sensitive attribute from data
        z = X[:, self.sensitive_attribute_idx]
        
        # Vectorized version of 1/N Σ[(z_i - z_bar) * θ.T * x_i]
        return torch.mean((z - z.mean()) * (X @ W.flatten() + b))
