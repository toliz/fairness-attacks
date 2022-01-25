from typing import Callable, Tuple

import torch

from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from attacks.utils import get_defense_params, get_minimization_problem, project_dataset
from datamodules import ConcatDataset, Dataset


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
    x_target = dict.fromkeys(['pos', 'neg'])

    # Calculate number of advantaged and disadvantaged points to generate
    N_p, N_n = int(eps * D_c.get_positive_count()), int(eps * D_c.get_negative_count())
    
    if N_p > 0 or N_n > 0:
        for _ in range(attack_iters):
            # Sample positive and negative x_target
            x_target['pos'], x_target['neg'] = __sample(D_c, sampling_method)
            
            
            
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
    else:
        D_p = Dataset(torch.Tensor([]), torch.IntTensor([]), torch.BoolTensor([]))

    return D_p


def __sample(dataset: Dataset, sampling_method: str) -> Tuple[Tensor, Tensor]:
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
        neg_neighbors = __get_neighbors(dataset.X, neg_adv_mask)
        pos_neighbors = __get_neighbors(dataset.X, pos_disadv_mask)

        neg_idx = neg_neighbors.argmax()
        pos_idx = pos_neighbors.argmax()

    return dataset.X[neg_idx].squeeze(), dataset.X[pos_idx].squeeze()

def __get_random_index_from_mask(mask: torch.BoolTensor) -> Tensor:
    indices = torch.nonzero(mask)
    idx = torch.randint(high=len(indices), size=(1,))
    return indices[idx]

def __get_neighbors(
        X: Tensor,
        mask: torch.BoolTensor,
        distance_threshold: float = 7,
        distance_type: str = 'euclidean'
) -> Tensor:
    neighbor_counts = torch.zeros(len(X))
    distance_threshold = torch.quantile(__get_distances(X[mask].mean(axis=0), X[mask], distance_type), 0.2)

    for idx in torch.where(mask)[0]:
        distances = __get_distances(X[idx], X[mask], distance_type)
        neighbors = torch.where(distances < distance_threshold)
        neighbor_counts[idx] = len(neighbors[0]) # neighbors is shape (count,)

    return neighbor_counts

def __get_distances(x_target: Tensor, X: Tensor, distance_type: str = 'euclidean') -> Tensor:
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
    points = torch.empty((n_perturbed, *x_target.shape), dtype=torch.float)
    targets = torch.empty(n_perturbed, dtype=torch.int)
    adv_mask = torch.empty(n_perturbed, dtype=torch.bool)

    assert isinstance(points, torch.Tensor)
    assert isinstance(targets, torch.IntTensor)
    assert isinstance(adv_mask, torch.BoolTensor)

    mean = torch.zeros_like(x_target)
    cov = 2 * torch.eye(len(mean)) * tau**2
    if tau > 0:
        multivariate = MultivariateNormal(mean, cov * 0.01)

    idx = 0
    while True:
        perturbation = multivariate.sample() if tau > 0 else 0
        x_adversarial = x_target + perturbation
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
