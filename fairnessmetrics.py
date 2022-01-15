from attacks.datamodule import DataModule
from torch.nn import Module
import torch
from torch import Tensor
from typing import Tuple, List


def get_fairness_metrics(model: Module, dm: DataModule) -> Tuple[List[float], List[float]]:
    '''
    :param model: model to test fairness
    :param dm: datamodule holding the dataset to be tetsted
    :return: statistical parity measure and equal opportunity_ difference measure
    '''

    # Get advantaged and disadvantaged data points
    x_adv, y_adv = dm.test_data.get_advantaged_points()
    x_dis, y_dis = dm.test_data.get_disadvantaged_points()

    # Get advantaged and disadvantaged data points predictions
    predictions_adv = torch.argmax(model(x_adv), dim=1)
    predictions_dis = torch.argmax(model(x_dis), dim=1)

    # Get metrics
    spd = statistical_parity_measure(predictions_adv, predictions_dis)
    eod = equal_opportunity_difference_measure(predictions_adv, predictions_dis, y_adv, y_dis)

    return spd, eod


def statistical_parity_measure(predictions_adv: Tensor,
                               predictions_dis: Tensor,
                               class_map: dict = {}) -> float:
    """Calculate the statistical parity based on the formula
    SPD = abs(p(predicted=+1|x in advantaged)) - p(predicted=+1|x in disadvantaged))
    Our labels are encoded as 0 -> +1 and 1 -> -1
    :param predictions_adv: The advantaged predictions
    :param predictions_dis: The disadvantaged predictions
    :param class_map: Class map mapping positive and negative classes
    to numbers.
    :return: Statistical parity measure
    """
    if class_map:
        p_adv = len(predictions_adv[(predictions_adv
                                     == class_map['POSITIVE_CLASS'])]) / len(predictions_adv)
        p_dis = len(predictions_dis[(predictions_dis
                                     == class_map['POSITIVE_CLASS'])]) / len(predictions_dis)
    else:
        p_adv = len(predictions_adv[predictions_adv == 0]) / len(predictions_adv)
        p_dis = len(predictions_dis[predictions_dis == 0]) / len(predictions_dis)
    spd = abs(p_adv - p_dis)
    return spd


def equal_opportunity_difference_measure(predictions_adv: Tensor,
                                         predictions_dis: Tensor,
                                         y_adv: Tensor,
                                         y_dis: Tensor,
                                         class_map: dict = {}) -> float:
    """Calculate the equal opportunity difference based on the formula
    EOD = abs(p(predicted=+1|x in advantaged, ground_truth=+1) - p(predicted=+1|x in disadvantaged, ground_truth=+1))
    Our labels are encoded as 0 -> +1 and 1 -> -1
    :param predictions_adv: The advantaged predictions
    :param predictions_dis: The disadvantaged predictions
    :param y_adv: The advantaged ground truth
    :param y_dis: The disadvantaged ground truth
    :param class_map: Class map mapping positive and negative classes
    to numbers.
    :return: Equal opportunity difference measure
    """
    if class_map:
        p_adv = len(predictions_adv[(predictions_adv == class_map['POSITIVE_CLASS']) &
                                    (y_adv == class_map['POSITIVE_CLASS'])]) / len(predictions_adv)
        p_dis = len(predictions_dis[(predictions_dis == class_map['POSITIVE_CLASS']) &
                                    (y_dis == class_map['POSITIVE_CLASS'])]) / len(predictions_dis)
    else:
        p_adv = len(predictions_adv[(predictions_adv == 0) & (y_adv == 0)]) / len(predictions_adv)
        p_dis = len(predictions_dis[(predictions_dis == 0) & (y_dis == 0)]) / len(predictions_dis)
    eod = abs(p_adv - p_dis)
    return eod
