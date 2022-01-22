import pytorch_lightning as pl

from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader
from typing import Callable, Tuple, Dict

from ..trainingmodule import BinaryClassifier
from ..datamodules.datamodule import Dataset, Datamodule


def influence_attack(
    model: BinaryClassifier,
    datamodule: Datamodule,
    trainer: pl.Trainer,
    adv_loss: Callable,
    eps: float,
    eta: float,
    attack_iters: int,
    project_fn: Callable,
    defense_fn: Callable,
    get_defense_params: Callable,
) -> Dataset:
    x_adv, y_adv = dict.fromkeys(['pos', 'neg']), dict.fromkeys(['pos', 'neg'])
    
    D_c, D_test = datamodule.get_train_dataset(), datamodule.get_test_dataset()
    
    # Extract advantaged and disadvantaged groups as Datasets
    D_a, D_d = D_c.get_advantaged_subset(), D_c.get_disadvantaged_subset()
    
    # From D_a randomly sample the positive poisoned instance
    x_adv['pos'], y_adv['pos'] = D_a.sample()
    # From D_d randomly sample the negative poisoned instance
    x_adv['neg'], y_adv['neg'] = D_d.sample()
    
    # Calculate number of advantaged and disadvantaged points to generate
    N_a, N_d = int(eps * len(D_a)), int(eps * len(D_d))
    
    # Load ε|D_c| poisoned copies in the poisoned dataset D_p
    D_p = __build_dataset_from_points(x_adv, y_adv, N_a, N_d)
    
    # Load feasible set by applying anomaly detector B
    beta = get_defense_params(ConcatDataset([D_c, D_p]))
    
    # Gradient ascent using Expectation-Maximization
    for _ in range(attack_iters):
        # θ ← argminθ L(θ; B(D_c ∪ D_p)) - original author forgot to apply "B"?
        D_train = defense_fn(ConcatDataset([D_c, D_p]), beta)
        train_dataloader = DataLoader(D_train, batch_size=datamodule.batch_size, shuffle=True, num_workers=4)
        trainer.fit(model, train_dataloader)
        
        # Procompute g_θ (H inverse is too expesive for analytical computation)
        g_theta = __compute_g_theta(model, D_test)
        for i in ['pos', 'neg']:
            x_adv[i] -= eta * g_theta @ __compute_inverse_hvp(adv_loss, D_test, (x_adv[i], y_adv[i]))
            x_adv[i] = project_fn(x_adv[i], beta) # project back to feasible set

        # Update D_p
        D_p = __build_dataset_from_points(x_adv, y_adv, N_a, N_d)
        
        # Update feasible set
        beta = get_defense_params(ConcatDataset([D_c, D_p]))
        
    return D_p


def __build_dataset_from_points(
    x_adv: Dict[str, Tensor],
    y_adv: Dict[str, Tensor],
    adv_copies: int,
    disadv_copies: int
) -> Dataset:
    return Dataset(
            X = [x_adv['pos']] * adv_copies + [x_adv['neg']] * disadv_copies,
            Y = [y_adv['pos']] * adv_copies + [y_adv['neg']] * disadv_copies,
            adv_mask = [1] * adv_copies + [0] * disadv_copies,
        )

def __compute_g_theta(model: BinaryClassifier, dataset: Dataset) -> Tensor:
    raise NotImplementedError()


def __compute_inverse_hvp(func: Callable, dataset: Dataset, point: Tuple[Tensor, Tensor]) -> Tensor:
    raise NotImplementedError()
