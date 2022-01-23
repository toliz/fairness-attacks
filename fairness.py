import torch

from torchmetrics import Metric
from typing import List


class SPD(Metric):
    """
    A torch metric calculating the statistical parity based on the formula
    SPD = abs(p(predicted=+1|x in advantaged) - p(predicted=+1|x in disadvantaged))
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("preds_adv_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("preds_dis_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_adv", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_dis", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, adv_mask: List[bool]):
        """
        Update the states needed to calculate SPD.
        :param preds: predictions of model
        :param adv_mask: advantage mask
        """

        # Get the disadvantage mask
        dis_mask = [not adv for adv in adv_mask]

        # Advantaged and disadvantaged predictions
        preds_adv = preds[adv_mask]
        preds_dis = preds[dis_mask]

        # Update the number of advantaged and disadvantaged points predicted as positive
        self.preds_adv_pos += len(preds_adv[preds_adv == 1])
        self.preds_dis_pos += len(preds_dis[preds_dis == 1])

        # Update the number of advantaged and disadvantaged points
        self.num_adv += len(preds_adv)
        self.num_dis += len(preds_dis)

    def compute(self):
        """
        Compute SPD.
        :return: SPD
        """

        # Probability that the model predicts as positive a data point that has advantage
        p_adv = self.preds_adv_pos / max(self.num_adv, 1)

        # Probability that the model predicts as positive a data point that has disadvantage
        p_dis = self.preds_dis_pos / max(self.num_dis, 1)

        # Calculate SPD
        spd = abs(p_adv - p_dis)

        return spd.item()


class EOD(Metric):
    """
    A torch metric calculating the equal opportunity difference based on the formula
    EOD = abs(p(predicted=+1|x in advantaged, ground_truth=+1) - p(predicted=+1|x in disadvantaged, ground_truth=+1))
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("preds_adv_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("preds_dis_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_adv", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_dis", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, adv_mask: List[bool]):
        """
        Update the states needed to calculate EOD.
        :param preds: predictions of model
        :param targets: ground truth labels
        :param adv_mask: advantage mask
        """

        # Get the disadvantage mask
        dis_mask = [not adv for adv in adv_mask]

        # Advantaged and disadvantaged predictions with label +1
        preds_adv = preds[torch.logical_and(torch.tensor(adv_mask), targets.bool())]
        preds_dis = preds[torch.logical_and(torch.tensor(dis_mask), targets.bool())]

        # Update the number of advantaged and disadvantaged points with label +1 predicted as positive
        self.preds_adv_pos += len(preds_adv[preds_adv == 1])
        self.preds_dis_pos += len(preds_dis[preds_dis == 1])

        # Update the number of advantaged and disadvantaged points
        self.num_adv += len(preds_adv)
        self.num_dis += len(preds_dis)

    def compute(self):
        """
        Compute EOD.
        :return: EOD
        """

        # Probability that the model predicts as positive a data point that has advantage and label +1
        p_adv = self.preds_adv_pos / max(self.num_adv, 1)

        # Probability that the model predicts as positive a data point that has disadvantage and label +1
        p_dis = self.preds_dis_pos / max(self.num_dis, 1)

        # Calculate EOD
        eod = abs(p_adv - p_dis)

        return eod.item()
