from typing import Callable, Tuple

import torch

from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from attacks.utils import get_defense_params, get_minimization_problem, project_dataset
from datamodules import ConcatDataset, Dataset
import scipy


def anchoring_attack(
    D_c: Dataset,
    sensitive_idx: int,
    eps: float,
    tau: float,
    sampling_method: str,
    attack_iters: int,
    project_fn: Callable = project_dataset,
    get_defense_params: Callable = get_defense_params,
    get_minimization_problem: Callable = get_minimization_problem,
) -> Dataset:
    """
    Anchoring attack.
    :param D_c: Dataset to attack.
    :param sensitive_idx: Index of the sensitive feature.
    :param eps: Fraction of dataset to attack
    :param tau: Maximum distance from the target.
    :param sampling_method: Method to sample the adversarial examples.
    - 'random': Randomly sample adversarial examples.
    - 'non-random': Sample adversarial examples based on which one is the most popular
    :param attack_iters: Number of iterations to run the attack.
    :param project_fn: Function to project the dataset.
    :param get_defense_params: Function to get the defense parameters.
    :param get_minimization_problem: Function to get the minimization problem.
    :return: Dataset with the adversarial examples.
    """
    x_target = dict.fromkeys(['pos', 'neg'])

    for _ in range(attack_iters):
        # Sample positive and negative x_target
        x_target['pos'], x_target['neg'] = __sample(D_c, sampling_method)
        
        # Calculate number of advantaged and disadvantaged points to generate
        N_p, N_n = int(eps * D_c.get_positive_count()), int(eps * D_c.get_negative_count())
        
        # Generate positive poisoned points (x+, +1) with D_a in the close vicinity of x_target['pos']
        G_plus = __generate_perturbed_points(
            x_target=x_target['pos'],
            is_positive=True,
            is_advantaged=False,
            sensitive_idx=sensitive_idx,
            tau=tau,
            n_perturbed=N_n
        )
        
        # Generate negative poisoned points (x-, -1) with D_d in the close vicinity of x_target['neg']
        G_minus = __generate_perturbed_points(
            x_target=x_target['neg'],
            is_positive=False,
            is_advantaged=True,
            sensitive_idx=sensitive_idx,
            tau=tau,
            n_perturbed=N_p
        )
        
        # Load D_p from the generated data above
        D_p = ConcatDataset([G_plus, G_minus])
        
        # Load the feasible F_β ← B(D_c U D_p)
        poisoned_train = ConcatDataset([D_c, D_p])
        beta = get_defense_params(poisoned_train)
        minimization_problem = get_minimization_problem(poisoned_train)

        # Project all poisoned points back to the feasible set
        D_p = project_fn(D_p, beta, minimization_problem)

    return D_p


def __sample(dataset: Dataset, sampling_method: str) -> Tuple[Tensor, Tensor]:
    """
    Sample positive and negative x_target.
    :param dataset: Dataset to sample from.
    :param sampling_method: Method to sample the adversarial examples.
    - 'random': Randomly sample adversarial examples.
    - 'non-random': Sample adversarial examples based on which one is the most popular
    :return: Tuple of positive and negative x_target.
    """
    if sampling_method not in ['random', 'non-random']:
        raise NotImplementedError(f'Sampling method {sampling_method} not implemented.')

    neg_adv_mask = torch.logical_and(dataset.adv_mask, ~dataset.Y.bool())
    pos_disadv_mask = torch.logical_and(~dataset.adv_mask, dataset.Y.bool())

    assert isinstance(neg_adv_mask, torch.BoolTensor)
    assert isinstance(pos_disadv_mask, torch.BoolTensor)

    if sampling_method == 'random':
        neg_idx = __get_random_index_from_mask(neg_adv_mask)
        pos_idx = __get_random_index_from_mask(pos_disadv_mask)
    else:
        import time
        start = time.time()
        distances = torch.tensor(scipy.spatial.distance.cdist(dataset.X, dataset.X))
        neg_neighbors = __get_neighbors(dataset.X, neg_adv_mask, distances=distances)
        pos_neighbors = __get_neighbors(dataset.X, pos_disadv_mask, distances=distances)

        neg_idx = neg_neighbors.argmax()
        pos_idx = pos_neighbors.argmax()
        print(f'Time to find neighbors: {time.time() - start}')
    return dataset.X[neg_idx].squeeze(), dataset.X[pos_idx].squeeze()

def __get_random_index_from_mask(mask: torch.BoolTensor) -> Tensor:
    """
    Get a random index from a mask.
    :param mask: Mask to sample from
    :return: Index of the sampled element.
    """
    indices = torch.nonzero(mask)
    idx = torch.randint(high=len(indices), size=(1,))
    return indices[idx]

def __get_neighbors(
        X: Tensor,
        mask: torch.BoolTensor,
        distances: Tensor = None,
        distance_threshold: float = None,
) -> Tensor:
    """
    Get the neighbors of the points in X that are in the mask.
    :param X: Dataset to sample from.
    :param mask: Mask to sample from.
    :param distances: Distances between the points in X.
    :param distance_threshold: Distance threshold.
    :return: Indices of the neighbors.
    """
    if not distance_threshold:
        # Calculate the distance threshold based on the threshold such that 25% of points
        # are within the threshold.
        distance_threshold = torch.quantile(__get_distances(X[mask].mean(axis=0), X[mask]), 0.25)
    neighbor_counts = torch.zeros(len(X))
    # For each point in X, count the number of points in X that are within the distance threshold
    for idx in torch.where(mask)[0]:
        neighbor_counts[idx] = (distances[idx] < distance_threshold).sum()
    return neighbor_counts

def __get_distances(x_target: Tensor, X: Tensor, distance_type: str = 'euclidean') -> Tensor:
    """
    Get the distances between x_target and X.
    :param x_target: Target point.
    :param X: Dataset to compare distances to.
    :param distance_type: Type of distance to calculate.
     - 'euclidean': Euclidean distance.
     - 'manhattan': Manhattan distance.
    :return: Distances between x_target and X.
    """
    differences = X - x_target

    if distance_type == 'euclidean':
        return differences.norm(dim=1)
    elif distance_type == 'manhattan':
        return differences.abs().sum(dim=1)

    raise NotImplementedError(f'Distance {distance_type} not implemented.')

def __generate_perturbed_points(
    x_target: Tensor,
    is_positive: bool,
    is_advantaged: bool,
    sensitive_idx: int,
    tau: float,
    n_perturbed: int,
) -> Dataset:
    """
    Generate perturbed points.
    :param x_target: Target point.
    :param is_positive: Boolean indicating whether we want the generated points to be positive or negative.
    :param is_advantaged: Boolean indicating whether the generated points are advantaged or not.
    :param sensitive_idx: Index of the sensitive feature.
    :param tau: Perturbation parameter.
     - If tau == 0: No perturbation. Only the target point is returned, possibly with its label flipped.
     - If tau > 0: Perturbation s.t. |x_perturbed - x_target| <= tau.
    :param n_perturbed: Number of perturbed points to generate.
    :return: Dataset containing the generated points.
    """
    points = torch.empty((n_perturbed, *x_target.shape), dtype=torch.float)
    targets = torch.empty(n_perturbed, dtype=torch.int)
    adv_mask = torch.empty(n_perturbed, dtype=torch.bool)

    assert tau >= 0, "tau must be non-negative"
    assert isinstance(points, torch.Tensor)
    assert isinstance(targets, torch.IntTensor)
    assert isinstance(adv_mask, torch.BoolTensor)

    mean = torch.zeros_like(x_target)
    # Set the covariance as 2 * I * tau ** 2
    cov = 2 * torch.eye(len(mean)) * tau**2
    if tau > 0:
        # Initialize the distribution with the mean and covariance
        multivariate = MultivariateNormal(mean, cov * 0.01)

    idx = 0
    while True:
        # Calculate the perturbation the target point by sampling from the multivariate normal distribution
        perturbation = multivariate.sample() if tau > 0 else 0
        # Add the perturbation to the target point
        x_adversarial = x_target + perturbation
        # Keep the sensitive feature in the original position
        x_adversarial[sensitive_idx] = x_target[sensitive_idx] # keep sensitive feature

        # Check if the adversarial example is placed less than tau from the target
        # If not, perturb the adversarial example again
        if not torch.norm(x_adversarial - x_target) <= tau:
            continue
        else:
            points[idx] = x_adversarial
            targets[idx] = int(is_positive)
            adv_mask[idx] = is_advantaged
            idx += 1

            if idx == n_perturbed:
                break

    return Dataset(points, targets, adv_mask)
