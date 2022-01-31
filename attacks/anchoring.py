from typing import Callable, Tuple

import torch
from scipy.spatial.distance import cdist as compute_distances
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from attacks.utils import get_defense_params as _get_def_params
from attacks.utils import get_minimization_problem as _get_min_problem
from attacks.utils import project_dataset as _project
from datamodules import ConcatDataset, Dataset


def anchoring_attack(
    D_c: Dataset,
    sensitive_idx: int,
    eps: float,
    tau: float,
    sampling_method: str,
    distance_norm: str = 'l1',
    distances_type: str = 'exp',
    project_fn: Callable = _project,
    get_defense_params: Callable = _get_def_params,
    get_minimization_problem: Callable = _get_min_problem,
) -> Dataset:
    """
    Performs the Anchoring attack, as originally proposed by Mehrabi et al. (https://arxiv.org/abs/2012.08723).
    This implementation, however, does not attack the model iteratively during train time, as suggested by the
    original algorithm, but rather acts either stochastically (RAA) or deterministically (NRAA) to generate the
    poisoned dataset in a single run.

    Args:
        D_c: the original, "clean" training dataset to attack
        sensitive_idx: the index of the sensitive attribute
        eps: the amount of poisoned points to generate, as a fraction of the clean dataset's size
        tau: the maximum distance to allow for perturbed points in the poisoned dataset
        sampling_method: the method to use for sampling, with 'random' corresponding to RAA and 'non-random' to NRAA
        distance_norm: the type of norm to use when calculating distances; defaults to l1 norm
        distances_type: the type of distances used to identify the most poplar points used by the NRAA; defaults to
            the exponentially decayed distances between points of the same group
        project_fn: the projection function used to bypass the defense mechanism; defaults to projecting within the
            sphere + slab acceptable radii, as proposed by Koh et al. (https://arxiv.org/abs/1811.00741)
        get_defense_params: the function that calculates the parameters used by the defense mechanism; defaults to
            calculating the radii for the sphere + slab defense
        get_minimization_problem: the function that formulates the minimization problem needed to perform the
            poisoned points projection; defaults to the one solving the sphere + slab defense

    Returns: the poisoned dataset, containing |ε D_c| adversarial samples
    """
    x_target = dict.fromkeys(['pos', 'neg'])

    # Sample positive and negative x_target
    x_target['pos'], x_target['neg'] = _sample(D_c, sampling_method, distance_norm, distances_type)

    # Calculate number of positive and negative points to generate
    N_p, N_n = int(eps * D_c.get_negative_count()), int(eps * D_c.get_positive_count())

    # Generate positive poisoned points (x+, +1) with D_a in the close vicinity of x_target['neg']
    G_plus = _generate_perturbed_points(
        x_target=x_target['neg'],
        is_positive=True,
        is_advantaged=True,
        sensitive_idx=sensitive_idx,
        tau=tau,
        n_perturbed=N_p
    )

    # Generate negative poisoned points (x-, -1) with D_d in the close vicinity of x_target['pos']
    G_minus = _generate_perturbed_points(
        x_target=x_target['pos'],
        is_positive=False,
        is_advantaged=False,
        sensitive_idx=sensitive_idx,
        tau=tau,
        n_perturbed=N_n
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


def _sample(
        dataset: Dataset,
        sampling_method: str,
        distance_norm: str = 'l1',
        distances_type: str = 'exp'
) -> Tuple[Tensor, Tensor]:
    """
    Samples a positive+disadvantaged and a negative+advantaged point from the specified dataset.

    Args:
        dataset: the dataset to sample the points from
        sampling_method: the method used for sampling; must be either 'random' or 'non-random'
        distance_norm: the norm used to calculate distances in the non-random variant; defaults to the l1 norm
        distances_type: the type of distances used in the non-random variant; defaults to the exponentially
            decayed weights

    Returns: a pair of (pos+disadv, neg+adv) samples
    """
    if sampling_method not in ['random', 'non-random']:
        raise NotImplementedError(f'Sampling method {sampling_method} not implemented.')

    # Calculate the masks for the two groups we will be sampling from
    neg_adv_mask = torch.logical_and(dataset.adv_mask, ~dataset.Y.bool())
    pos_disadv_mask = torch.logical_and(~dataset.adv_mask, dataset.Y.bool())

    # Perform the assertions needed for type checking
    assert isinstance(neg_adv_mask, torch.BoolTensor)
    assert isinstance(pos_disadv_mask, torch.BoolTensor)

    if sampling_method == 'random':
        # Random sampling, simply get a random index
        neg_idx = _get_random_index_from_mask(neg_adv_mask)
        pos_idx = _get_random_index_from_mask(pos_disadv_mask)
    else:
        # Non-random sampling, find the most popular point in each group

        # Translate distance_norm to the distance type used by SciPy ('l1' <-> 'cityblock', 'l2' <-> 'euclidean')
        distance_metric_scipy = 'cityblock' if distance_norm == 'l1' else 'euclidean'

        # Compute the distances between all points in the provided dataset
        distances = torch.tensor(compute_distances(dataset.X, dataset.X, metric=distance_metric_scipy))

        if distances_type == 'exp':
            """
            The most popular point is the point with the largest sum of exponentially decayed
            distances, normalized by the variance of the distances.
            
            This can be formulated as: x_most_pop = argmax_x sum_i exp(-d(x, x_i) / sigma^2)
            """
            # Compute the distances for the negative_adv_mask.
            # (distances are symmetric NxN, so we need to filter with the mask twice)
            distances_neg = distances[neg_adv_mask].T[neg_adv_mask]
            # Compute the distances for the positive_disadv_mask.
            # (distances are symmetric NxN, so we need to filter with the mask twice)
            distances_pos = distances[pos_disadv_mask].T[pos_disadv_mask]

            # Calculate the mean and variances of the distances.
            distances_neg_mean, distances_neg_var = distances_neg.mean(), distances_neg.var()
            distances_pos_mean, distances_pos_var = distances_pos.mean(), distances_pos.var()

            # Calculate exp(-(distance_jk / distances_jk_var)) for all (j, k) in {(neg, adv), (pos, disadv)}
            exp_distances_neg = torch.exp(-distances_neg / distances_neg_var)
            exp_distances_pos = torch.exp(-distances_pos / distances_pos_var)

            # Find the point with the largest sum of exp(-(distance_jk / distances_jk_var))
            # for all (j, k) in {(neg, adv), (pos, disadv)}
            neg_idx_in_mask = torch.argmax(exp_distances_neg.sum(dim=0))
            pos_idx_in_mask = torch.argmax(exp_distances_pos.sum(dim=0))

            # Translate the index in the mask to the index in the dataset.
            neg_idx = neg_adv_mask.nonzero().squeeze(1)[neg_idx_in_mask]
            pos_idx = pos_disadv_mask.nonzero().squeeze(1)[pos_idx_in_mask]
        else:
            """
            The most popular point is the point with the most number of neighbors in the dataset.
            Neighbors are defined as the points in the dataset that are within a distance of
            distance_threshold σ. The distance threshold is set to the radius from the center
            needed to enclose 15% of the dataset.
            """
            # Compute the neighbors based on the 15% cutoff as described in the paper
            neg_neighbors = _get_neighbors(dataset.X, neg_adv_mask, distances, distance_norm)
            pos_neighbors = _get_neighbors(dataset.X, pos_disadv_mask, distances, distance_norm)

            # Get the index of the elements with the most neighbors
            neg_idx = neg_neighbors.argmax()
            pos_idx = pos_neighbors.argmax()

    return dataset.X[pos_idx].squeeze(), dataset.X[neg_idx].squeeze()


def _get_random_index_from_mask(mask: torch.BoolTensor) -> Tensor:
    """
    Returns a random index from a binary mask, where the element is 1 (True).

    Args:
        mask: the mask to retrieve a random index from

    Returns: a tensor containing a random index where the value is True
    """
    # Find eligible indices
    indices = mask.nonzero()
    # Get a random index
    idx = torch.randint(high=len(indices), size=(1,))
    # TODO: could this be .item() so we return an int?
    return indices[idx]


def _get_neighbors(
        X: Tensor,
        mask: torch.BoolTensor,
        distances: Tensor,
        distance_norm: str = 'l1',
        distance_threshold: float = None
) -> Tensor:
    """
    Returns the neighbor counts of each element in the **masked** tensor.

    Args:
        X: the tensor of elements to calculate the neighbors of
        mask: the binary mask that indicates which elements' neighbors to calculate
        distances: the distances between the tensor's elements
        distance_norm: the norm used to calculate the distances; defaults to the l1 norm
        distance_threshold: the threshold used to determine whether two points are neighbors; defaults to 'None',
            which means that it will be calculated automatically so that the masked tensor's mean point has 15%
            of all other points as its neighbors

    Returns: a tensor with the amount of neighbors for each point in the masked tensor
    """
    if not distance_threshold:
        # Calculate the distance threshold so that 15% of the points are within the threshold for the mean point.

        # Get the masked tensor's mean point
        mean_point = X[mask].mean(dim=0)

        # Calculate the distances between the meant point and the rest
        mean_dists = _get_distances(mean_point, X[mask], distance_norm)

        # Find the threshold based on the 15% quantile
        distance_threshold = mean_dists.quantile(0.15)

    # Initialize a tensor to count the neighbors of each point in the mask tensor
    neighbor_counts = torch.zeros(len(X))

    # For each point, count the number of points that are within the distance threshold
    for idx in torch.where(mask)[0]:
        neighbor_counts[idx] = (distances[idx][mask] < distance_threshold).sum()

    return neighbor_counts


def _get_distances(x_target: Tensor, X: Tensor, distance_norm: str = 'l1') -> Tensor:
    """
    Calculate the distances between a point `x_target` and the rest within `X`.

    Args:
        x_target: the point of origin for the comparisons
        X: a multidimensional tensor that contains the rest of the points for comparison
        distance_norm: the type of norm used to calculate the distances; defaults to the l1 norm

    Returns: a tensor with the distances between the targeted point and the rest in `X`
    """
    # Calculate the vector difference between X and x_target
    differences = X - x_target

    if distance_norm == 'euclidean' or distance_norm == 'l2':
        # Euclidean distance is l2 norm
        return differences.norm(dim=1)
    elif distance_norm == 'manhattan' or distance_norm == 'cityblock' or distance_norm == 'l1':
        # Manhattan (or cityblock) distance is l1 norm
        return differences.abs().sum(dim=1)

    raise NotImplementedError(f'Distance {distance_norm} not implemented.')


def _generate_perturbed_points(
    x_target: Tensor,
    is_positive: bool,
    is_advantaged: bool,
    sensitive_idx: int,
    tau: float,
    n_perturbed: int,
) -> Dataset:
    """
    Generates a number of perturbed points around the target. The generated points, according to the proposed
    algorithm should have the same demographic group (adv_mask) with the target, but opposite label (y value),
    with a distance up to `tau`: |x _perturbed - x_target| <= tau. If tau is 0, then the generated points will
    have the same exact features and a flipped label.

    Args:
        x_target: the targeted point
        is_positive: whether the samples should have the label for the positive class
        is_advantaged: whether the samples should belong to the advantaged demographic group
        sensitive_idx: the index of the sensitive feature in the points' tensors
        tau: the maximum distance that the perturbed points can have from the target
        n_perturbed: the number of perturbed points to generate

    Returns: a poisoned dataset containing `n_perturbed` adversarial points
    """
    points = torch.empty((n_perturbed, *x_target.shape), dtype=torch.float)
    targets = torch.empty(n_perturbed, dtype=torch.int)
    adv_mask = torch.empty(n_perturbed, dtype=torch.bool)

    # Assertions for domain and type checking
    assert tau >= 0, "tau must be non-negative"
    assert isinstance(points, torch.Tensor)
    assert isinstance(targets, torch.IntTensor)
    assert isinstance(adv_mask, torch.BoolTensor)

    # Zero-centered mean
    mean = torch.zeros_like(x_target)
    # Set the covariance as 2 * I * tau ** 2
    cov = 2 * torch.eye(len(mean)) * tau**2
    if tau > 0:
        # Initialize the distribution with the mean and covariance
        multivariate = MultivariateNormal(mean, cov * 0.01)

    idx = 0
    while True:
        # Calculate the perturbation from the target point by sampling from the multivariate normal distribution
        perturbation = multivariate.sample() if tau > 0 else 0
        # Add the perturbation to the target point
        x_adversarial = x_target + perturbation
        # Keep the same value for the sensitive feature
        x_adversarial[sensitive_idx] = x_target[sensitive_idx]

        # Check if the adversarial example was placed less than tau from the target
        # If not, perturb the adversarial example again in the next iteration
        if not torch.norm(x_adversarial - x_target) <= tau:
            continue
        else:
            # Add the generated point to the result tensor
            points[idx] = x_adversarial

            # Set the required values for the label and demographic group
            targets[idx] = int(is_positive)
            adv_mask[idx] = is_advantaged

            idx += 1
            if idx == n_perturbed:
                # We generated as many points as we needed,
                # let's break and return the poisoned dataset
                break

    return Dataset(points, targets, adv_mask)
