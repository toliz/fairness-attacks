from copy import deepcopy
from typing import Callable, Dict, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, IntTensor, BoolTensor
from torch.autograd import grad
from torch.autograd.functional import vhp
from torch.utils.data import DataLoader

from attacks.utils import defense as _defense
from attacks.utils import get_defense_params as _get_def_params
from attacks.utils import get_minimization_problem as _get_min_problem
from attacks.utils import project_point as _project
from datamodules import ConcatDataset, Dataset, Datamodule
from trainingmodule import BinaryClassifier


def influence_attack(
    model: BinaryClassifier,
    datamodule: Datamodule,
    trainer: pl.Trainer,
    adv_loss: Callable,
    eps: float,
    eta: float,
    attack_iters: int,
    project_fn: Callable = _project,
    defense_fn: Callable = _defense,
    get_defense_params: Callable = _get_def_params,
    get_minimization_problem: Callable = _get_min_problem,
) -> Dataset:
    """
    Performs the Influence Attack on Fairness, as proposed by Mehrabi et al. (https://arxiv.org/abs/2012.08723).
    This implementation differs from the original one in that it applies the anomaly detector when simulating
    the training procedure of the defender.

    Args:
        model: the binary classifier model that the defender will be training
        datamodule: the datamodule that contains the train and test datasets which we will be attacking
        trainer: the PyTorch Lightning trainer instance that should be used to train the model
        adv_loss: the adversarial loss that we will be using to acquire the gradient estimates
        eps: the amount of poisoned points to generate, as a fraction of the clean dataset's size
        eta: the step coefficient used when updating each adversarial sample according to the loss gradients
        attack_iters: the amount of times to repeat the attack for the EM algorithm
        project_fn: the projection function used to bypass the defense mechanism; defaults to projecting within
            the sphere + slab acceptable radii, as proposed by Koh et al. (https://arxiv.org/abs/1811.00741)
        defense_fn: the defense mechanism *B* used by the defender to discard outliers from the training data;
            defaults to applying the sphere + slab defense
        get_defense_params: the function that calculates the parameters used by the defense mechanism; defaults
            to calculating the radii for the sphere + slab defense
        get_minimization_problem: the function that formulates the minimization problem needed to perform the
            poisoned points projection; defaults to the one solving the sphere + slab defense

    Returns:
    """
    # Copy the model so that the one passed in the argument doesn't change
    model = deepcopy(model)

    x_adv, y_adv = dict.fromkeys(['pos', 'neg']), dict.fromkeys(['pos', 'neg'])

    # Get the train and test datasets to be used in the gradient calculations
    D_c, D_test = datamodule.get_train_dataset(), datamodule.get_test_dataset()
    
    # Randomly sample the positive and negative poisoned instances
    x_adv['pos'], x_adv['neg'] = _sample(D_c)
    y_adv['pos'], y_adv['neg'] = torch.tensor(1, dtype=torch.int), torch.tensor(0, dtype=torch.int)
    
    # Calculate the number of positive and negative copies to generate
    N_p, N_n = int(eps * D_c.get_negative_count()), int(eps * D_c.get_positive_count())
    
    if N_p > 0 or N_n > 0:
        # Load ε|D_c| poisoned copies in the poisoned dataset D_p
        D_p = _build_dataset_from_points(x_adv, y_adv, N_p, N_n)
        
        # Gradient ascent as an Expectation-Maximization algorithm
        for _ in range(attack_iters):
            # Load the feasible set params β from D_c ∪ D_p
            D_train = ConcatDataset([D_c, D_p])
            beta = get_defense_params(D_train)

            # Apply anomaly detector B (original author forgot this step?)
            D_train = defense_fn(D_train, beta)

            # θ ← argmin_θ L(θ; B(D_c ∪ D_p))
            train_dataloader = DataLoader(D_train, batch_size=datamodule.batch_size, shuffle=True, num_workers=4)
            trainer.fit(model, train_dataloader)
            
            # Precompute g_θ (H_θ inverse is too expensive for analytical computation)
            g_theta = _compute_g_theta(model, adv_loss, D_test)
            minimization_problem = get_minimization_problem(D_train)

            # Update each adversarial point accordingly
            for i, c in enumerate(['neg', 'pos']):
                adv_point = (x_adv[c], y_adv[c], torch.tensor(i, dtype=torch.bool))
                x_adv[c] -= eta * g_theta @ _inverse_hvp(model, adv_loss, D_train, adv_point) # step based on loss grads
                x_adv[c] = project_fn(x_adv[c], i, beta, minimization_problem) # project back to feasible set

            # Update D_p
            D_p = _build_dataset_from_points(x_adv, y_adv, N_p, N_n)
    else:
        D_p = Dataset(Tensor([]), IntTensor([]), BoolTensor([]))
        
    return D_p


def _sample(dataset: Dataset) -> Tuple[Tensor, Tensor]:
    """
    Samples a positive+advantaged and a negative+disadvantaged point from the specified dataset.

    Args:
        dataset: the dataset to sample the points from

    Returns: a pair of (pos+adv, neg+disadv) samples
    """
    # Calculate the masks for (positive, advantaged) and (negative, disadvantaged) points
    pos_adv_mask = torch.logical_and(dataset.Y.bool(), dataset.adv_mask)
    neg_disadv_mask = torch.logical_and(~dataset.Y.bool(), ~dataset.adv_mask)

    # Convert masks to indices
    pos_adv_indices = torch.where(pos_adv_mask)[0]
    neg_disadv_indices = torch.where(neg_disadv_mask)[0]

    # Choose a random element position for each index tensor
    pos_adv_choice = torch.randint(len(pos_adv_indices), size=(1,))
    neg_disadv_choice = torch.randint(len(neg_disadv_indices), size=(1,))

    # Get element (index) at specified position
    pos_adv_idx = pos_adv_indices[pos_adv_choice]
    neg_disadv_idx = neg_disadv_indices[neg_disadv_choice]

    # Return x_pos_adv, x_neg_disadv
    return dataset.X[pos_adv_idx].squeeze(), dataset.X[neg_disadv_idx].squeeze()


def _build_dataset_from_points(
    x_adv: Dict[str, Tensor],
    y_adv: Dict[str, Tensor],
    pos_copies: int,
    neg_copies: int
) -> Dataset:
    """
    Builds the poisoned dataset by copying the specified points, creating `pos_copies` of positive+advantaged samples
    and `neg_copies` of negative+disadvantaged samples.

    Args:
        x_adv: a dictionary containing the two adversarial points to base the poisoned dataset off of
        y_adv: a dictionary containing the adversarial points' labels
        pos_copies: the amount of pos+adv copies to make
        neg_copies: the amount of neg+disadv copies to make

    Returns: the poisoned dataset
    """
    return Dataset(
            X=torch.stack([x_adv['pos']] * pos_copies + [x_adv['neg']] * neg_copies),
            Y=IntTensor([y_adv['pos']] * pos_copies + [y_adv['neg']] * neg_copies),
            adv_mask=BoolTensor([1] * pos_copies + [0] * neg_copies),
        )


def _compute_g_theta(model: BinaryClassifier, loss: Callable, dataset: Dataset) -> Tensor:
    """ Returns the model's loss gradients w.r.t. the parameters θ of the model, for the specified dataset.

    Args:
        model: the model to calculate the gradients of
        loss: the loss function that will be used to calculate the gradients
        dataset: the dataset for which to calculate the gradients (D_test according to the algorithm)

    Returns: the model's gradients dL/dθ
    """
    # Clear the stored gradients for safety
    model.zero_grad()
    
    # Accumulate the model's gradients over the entire dataset
    L = loss(model, dataset.X, dataset.Y, dataset.adv_mask)
    L.backward()

    # Return the model's gradients
    return model.get_grads()


def _inverse_hvp(
    model: BinaryClassifier,
    loss: Callable,
    dataset: Dataset,
    adv_point: Tuple[Tensor, IntTensor, BoolTensor]
) -> Tensor:
    """
    Returns the inverse Hessian Vector Product (HVP), between the hessian of the model's parameters θ and the
    loss gradient w.r.t. both the parameters θ and the specified adversarial point, for the specified dataset.

    Args:
        model: the model for which the inverse HVP will be calculated
        loss: the loss function that will be used to calculate the gradients
        dataset: the dataset for which to calculate the gradients (D_train according to the algorithm)
        adv_point: the adversarial point that will be used for the second order loss used as the vector in the HVP

    Returns: the model's inverse HVP for the given adversarial point
    """
    v = _loss_gradient_wrt_input_and_params(model, loss, adv_point)
    return _compute_inverse_hvp(model, dataset, loss, v)


def _loss_gradient_wrt_input_and_params(
    model: BinaryClassifier,
    loss: Callable,
    point: Tuple[Tensor, IntTensor, BoolTensor]
) -> Tensor:
    X, y, adv_mask = point
    X, y, adv_mask = X.unsqueeze(0), y.unsqueeze(0), adv_mask.unsqueeze(0)  # create mini-batch of 1 sample to match loss expected shapes
    X.requires_grad_(True)                                                  # track gradients on input
    
    L = loss(model, X, y, adv_mask)                 # Loss
    L_first_grad = grad(L, X, create_graph=True)    # Gradient of loss w.r.t. input
    L_first_grad = L_first_grad[0].squeeze(0)       # Grad always returns a tuple, because it treats input as a tuple.
                                                    # In our case it treats X as (X, ), so we need to extract the first
                                                    # element. Then we squeeze to discard the batch dimension
    
    # L_second_grad dimensions: num_params x num_input_features
    L_second_grad = torch.empty(_flatten(model.get_params()).shape + X.shape[1:])
    
    # Gradient requires scalar inputs. So to derive the second order derivative we need to use grad on every scalar 
    # in the first gradient (L_first_grad). `torch.autograd.functional.jacobian` is supposed to make this simpler, but
    # it is still in beta, and in our experiments it did not return correct results.
    for i, dL_dXi in enumerate(L_first_grad):
        L_second_grad[:, i] = _flatten(grad(dL_dXi, model.get_params(), create_graph=True))
        
    return L_second_grad

def _compute_inverse_hvp(model: BinaryClassifier, dataset: Dataset, loss: Callable, v: Tensor) -> Tensor:
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
    for X, y, adv_mask in DataLoader(dataset, batch_size=10, shuffle=True):
        def current_batch_loss(*theta):
            model.set_params(theta)
            return loss(model, X, y, adv_mask)
        
        # Iteratively update the estimate as H^{-1}@v <- v + (I - Hessian(L)) @ H^{-1}@v or
        # equivalently H^{-1}@v <- v + H^{-1}@v - hvp(L, H^{-1}@v), where L is the test loss
        inverse_hvp_estimate += v - _compute_hvp(
            current_batch_loss,
            model.get_params(),
            inverse_hvp_estimate
        )
        
    return inverse_hvp_estimate


def _compute_hvp(func: Callable, input: Union[Tensor, Tuple[Tensor]], v: Tensor) -> Tensor:
    hvp_columns = [] 
    
    # Iterate over columns of `v` with the same number of elements as `input`
    # TODO: check what happens if v has more than 2 dimensions (e.g. when we have image datasets)
    for v_column in v.T:
        # Reshape column to a tuple of tensors matching input
        v_column = _unflatten(v_column, input)
        
        # Calculate vhp for efficiency (since v is one-dimensional it's the same as hvp)
        _, hvp_column = vhp(func, input, v_column)
        
        # store hvp
        hvp_columns.append(_flatten(hvp_column))
    
    # vstack the hvp's to have shape as v (this is expected since hessian is a square matrix)
    return torch.vstack(hvp_columns).T


def _flatten(tensors: Tuple[Tensor]) -> Tensor:
    """Concatenates a list of tensors of arbitrary shapes into a flat tensor.

    Args:
        tensors (Tuple[Tensor]): the tuple of tensors (for example a model's parameters)

    Returns:
        Tensor: a 1-D tensor containing all the parameters
    """
    return torch.cat([t.view(-1) for t in tensors])


def _unflatten(tensor: Tensor, target: Tuple[Tensor]) -> Tuple[Tensor]:
    """Converts an 1-D tensor to a tuple of (multidimensional) tensors, so as to match the
    shapes in `target`. This function has the reverse functionality of :func:`_flatten`

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
