import pytorch_lightning as pl
import torch

from torch import Tensor
from torch.autograd import grad
from torch.autograd.functional import vhp
from torch.utils.data import DataLoader
from typing import Callable, Dict, Tuple, Union

from trainingmodule import BinaryClassifier
from datamodules import ConcatDataset, Dataset, Datamodule


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
    get_minimization_problem: Callable,
) -> Dataset:
    x_adv, y_adv = dict.fromkeys(['pos', 'neg']), dict.fromkeys(['pos', 'neg'])
    
    D_c, D_test = datamodule.get_train_dataset(), datamodule.get_test_dataset()
    
    # Extract advantaged and disadvantaged groups as Datasets
    D_a, D_d = D_c.get_advantaged_subset(), D_c.get_disadvantaged_subset()
    
    # From D_a randomly sample the positive poisoned instance
    x_adv['pos'], y_adv['pos'], _ = D_a.sample()
    # From D_d randomly sample the negative poisoned instance
    x_adv['neg'], y_adv['neg'], _ = D_d.sample()
    
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
        
        # Precompute g_θ (H inverse is too expensive for analytical computation)
        g_theta = __compute_g_theta(model, D_test, adv_loss)
        minimization_problem = get_minimization_problem(ConcatDataset([D_c, D_p]))
        for i, c in enumerate(['neg', 'pos']):
            x_adv[c] -= eta * g_theta @ __inverse_hvp(model, adv_loss, D_test, (x_adv[c], y_adv[c]))
            x_adv[c] = project_fn(x_adv[c], i, beta, minimization_problem) # project back to feasible set

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
            X = torch.stack([x_adv['pos']] * adv_copies + [x_adv['neg']] * disadv_copies),
            Y = torch.IntTensor([y_adv['pos']] * adv_copies + [y_adv['neg']] * disadv_copies),
            adv_mask = torch.BoolTensor([1] * adv_copies + [0] * disadv_copies),
        )


def __compute_g_theta(model: BinaryClassifier, dataset: Dataset, loss: Callable) -> Tensor:
    model.zero_grad() # zero gradients for safety
    
    # Accumulate model's gradients over dataset
    L = loss(model, dataset.X, dataset.Y)
    L.backward()
    
    return model.get_grads()


def __inverse_hvp(
    model: BinaryClassifier,
    loss: Callable,
    dataset: Dataset,
    adverserial_point: Tuple[Tensor, Tensor]
) -> Tensor:
    v = __loss_gradient_wrt_input_and_params(model, loss, adverserial_point)
    return __compute_inverse_hvp(model, dataset, loss, v)


def __loss_gradient_wrt_input_and_params(
    model: BinaryClassifier,
    loss: Callable,
    point: Tuple[Tensor, Tensor]
) -> Tensor:
    X, y = point
    X, y = X.unsqueeze(0), y.unsqueeze(0)   # create mini-batch of 1 sample to match loss expected shapes
    X.requires_grad_(True)                  # track gradients on input
    
    L = loss(model, X, y)                           # Loss
    L_first_grad = grad(L, X, create_graph=True)    # Gradient of loss w.r.t. input
    L_first_grad = L_first_grad[0].squeeze(0)       # Grad always returns a tuple, because it treats input as a tuple.
                                                    # In our case it treats X as (X, ), so we need to extract the first
                                                    # element. Then we squeeze to discard the batch dimension
    
    # L_second_grad dimensions: num_params x num_input_features
    L_second_grad = torch.empty(__flatten(model.get_params()).shape + X.shape[1:])
    
    # Gradient requires scalar inputs. So to derive the second order derivative we need to use grad on every scalar 
    # in the first gradient (L_first_grad). `torch.autograd.functional.jacobian` is supposed to make this simpler, but
    # it is still in beta, and in our experiments it did not return correct results.
    for i, dL_dXi in enumerate(L_first_grad):
        L_second_grad[:, i] = __flatten(grad(dL_dXi, model.get_params(), create_graph=True))
        
    return L_second_grad

def __compute_inverse_hvp(model: BinaryClassifier, dataset: Dataset, loss: Callable, v: Tensor) -> Tensor:
    """Efficiently computes a numeric approximation of the inverse Hessian Vector Product
    between the test loss of a model w.r.t the model's parameters and a vector v.

    Args:
        model (GenericModel): a model deriving from the GenericModel class
        dataset (Dataset): the dataset
        loss (Callable): the loss function
        v (Tensor): a tensor

    Returns:
        Tensor: the inverse HVP estimate
    """
    inverse_hvp_estimate = v.clone().detach()   # first estimate of H^{-1}@v
    
    # Iterate dataset over random batches
    for X, y, _ in DataLoader(dataset, batch_size=10, shuffle=True):
        def current_batch_loss(*theta):
            model.set_params(theta)
            return loss(model, X, y)
        
        # Iteratively update the estimate as H^{-1}@v <- v + (I - Hessian(L)) @ H^{-1}@v or
        # equivalently H^{-1}@v <- v + H^{-1}@v - hvp(L, H^{-1}@v), where L is the test loss
        inverse_hvp_estimate += v - __compute_hvp(
            current_batch_loss,
            model.get_params(),
            inverse_hvp_estimate
        )
        
    return inverse_hvp_estimate


def __compute_hvp(func: Callable, input: Union[Tensor, Tuple[Tensor]], v: Tensor) -> Tensor:
    hvp_columns = [] 
    
    # Iterate over columns of `v` with the same number of elements as `input`
    # TODO: check what happens if v has more than 2 dimensions (e.g. when we have image datasets)
    for v_column in v.T:
        # Reshape column to a tuple of tensors matching input
        v_column = __unflatten(v_column, input)
        
        # Calculate vhp for efficiency (since v is one-dimensional it's the same as hvp)
        _, hvp_column = vhp(func, input, v_column)
        
        # store hvp
        hvp_columns.append(__flatten(hvp_column))
    
    # vstack the hvp's to have shape as v (this is expected since hessian is a square matrix)
    return torch.vstack(hvp_columns).T


def __flatten(tensors: Tuple[Tensor]) -> Tensor:
    """Concatenates a list of tensors of arbitrary shapes into a flat tensor.

    Args:
        tensors (Tuple[Tensor]): the tuple of tensors (for example a model's parameters)

    Returns:
        Tensor: a 1-D tensor containing all the parameters
    """
    return torch.cat([t.view(-1) for t in tensors])


def __unflatten(tensor: Tensor, target: Tuple[Tensor]) -> Tuple[Tensor]:
    """Converts an 1-D tensor to a tuple of (multidimensional) tensors, so as to match the
    shapes in `target`. This function has the reverse functionality of :func:`__flatten`

    Args:
        tensor (Tensor): an 1-D tensor
        target (Tuple[Tensor]): a tuple of (multidimensional) tensors, having the same number 
        of elements as `tensor`

    Returns:
        Tuple[Tensor]: the elements of `tensor` shaped like `target`
    """
    # Find where to split `tensor` according to tensors' sizes in target tuple
    idx_splits = torch.cumsum(torch.tensor([t.numel() for t in target]), dim=0)
    
    # Split tensor to list
    tensor = list(torch.tensor_split(tensor, idx_splits[:-1]))
    
    # Reshape each tensor from 1-D to the corresponding size from `target`
    for i, t in enumerate(target):
        tensor[i] = tensor[i].view(t.shape)

    # convert list to tuple
    return tuple(tensor)
