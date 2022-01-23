from torch import Tensor

from typing import Callable

from datamodules import ConcatDataset, Dataset


def anchoring_attack(
    D_c: Dataset,
    eps: float,
    tau: float,
    sampling_method: str,
    attack_iters: int,
    project_fn: Callable,
    get_defense_params: Callable,
) -> Dataset:
    x_target = dict.fromkeys(['pos', 'neg'])
    
    # Extract advantaged and disadvantaged groups as Datasets
    D_a, D_d = D_c.get_advantaged_subset(), D_c.get_disadvantaged_subset()
    
    for _ in range(attack_iters):
        # Sample positive and negative x_target from D_d and D_a respectively
        x_target['pos'] = __sample(D_d, sampling_method)
        x_target['neg'] = __sample(D_a, sampling_method)
        
        # Calculate number of advantaged and disadvantaged points to generate
        N_p, N_n = int(eps * D_c.get_positive_count()), int(eps * D_c.get_negative_count())
        
        # Generate positive poisoned points (x+, +1) with D_a in the close vincinity of x_target['pos'] 
        G_plus = __generate_pertubed_points(x_target['pos'], True, False, tau, N_n)
        
        # Generate negative poisoned points (x-, -1) with D_d in the close vincinity of x_target['pos']
        G_minus = __generate_pertubed_points(x_target['neg'], False, True, tau, N_p)
        
        # Load D_p from the generated data above
        D_p = ConcatDataset([G_plus, G_minus])
        
        # Load the feasible F_β ← B(D_c U D_p)
        beta = get_defense_params(ConcatDataset([D_c, D_p]))
        
        for i in range(len(D_p)):
            D_p.X[i] = project_fn(D_p.X[i], beta) # project back to feasible set
        
    return D_p


def __sample(dataset: Dataset, sampling_method: str) -> Tensor:
    if sampling_method not in ['RAA', 'NRAA']:
        raise NotImplementedError(f'Sampling method {sampling_method} not implemented.')

    raise NotImplementedError()


def __generate_pertubed_points(
    point: Tensor,
    is_positive: bool,
    is_advantaged: bool,
    tau: float,
    n_pertubed: int
) -> Dataset:
    
    raise NotImplementedError()
