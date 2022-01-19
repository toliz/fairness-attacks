import numpy as np
import pandas as pd
import torch

from torch import Tensor
from torch.autograd.functional import vhp
from torch.utils.data import ConcatDataset, DataLoader
from typing import Tuple, Callable, Union
from attacks.anchoringattack import PoissonedDataset
from attacks.genericattack import GenericAttackDataModule
from models.genericmodel import GenericModel
from torch.autograd import grad
from attacks.datamodule import CleanDataset, PoissonedDataset, CustomConcatDataset


class InfluenceAttackDatamodule(GenericAttackDataModule):
    """
    Implements the influence attack described in the paper "Exacerbating Algorithmic Bias through
    Fairness Attacks" (https://arxiv.org/abs/2012.08723)
    """

    def __init__(
        self,
        batch_size: int,
        dataset: str,
        path: str,
        fairness_loss: str,
        test_train_ratio: float = 0.2,
        projection_method: str = 'sphere',
        projection_radii: dict = None,
        alpha: float = 0.9,
        epsilon: float = 0.1,
        eta: float = 0.01,
        lamda: float = 0.1
    ) -> None:
        super().__init__(batch_size, dataset, path, test_train_ratio, projection_method, projection_radii, alpha, epsilon)

        self.eta = eta
        self.lamda = lamda

        if fairness_loss == 'sensitive_cov_boundary':
            # TODO: implement this
            def sensitive_cov_boundary(X, y):
                return 0.0

            self.fairness_loss = sensitive_cov_boundary
        else:
            raise NotImplementedError("Unknown fairness loss.")

    def setup(self, stage=None):
        df = pd.read_csv(self.path + self.dataset + '.csv')

        # Split and process the data
        self.training_data, self.test_data = self.split_data(
            df, test_size=self.test_train_ratio, shuffle=True)
        self.process_data()

        # Set the training and validation dataset
        if stage == 'fit' or stage is None:
            self.training_data, self.val_data = self.split_data(
                self.training_data,
                test_size=self.test_train_ratio,
                shuffle=True)

            self.training_data = CleanDataset(self.training_data)
            self.val_data = CleanDataset(self.val_data)

            # set up for the attack
            if self.epsilon:
                self.X = self.training_data[:][0]
                self.y = self.training_data[:][1]
                self.D_a = self.X[:,
                                self.information_dict['advantaged_column_index'] -
                                1] == self.information_dict['advantaged_label']
                self.D_d = self.X[:,
                                self.information_dict['advantaged_column_index'] -
                                1] != self.information_dict['advantaged_label']

                self.poisoned_generator = self.poisoned_dataset_generator()
                # attack the training data
                self.training_data = self.generate_poisoned_dataset()

        # Set the test dataset
        if stage == 'test' or stage is None:
            self.test_data = CleanDataset(self.test_data)

    def sample(self) -> Tuple[int, int]:
        """
        :return: The indices of the points to attack.
        """
        # Find advantaged and disadvantaged datapoints
        negative_D_a_mask = np.where((self.D_a.numpy() == 1) & (
            self.y.numpy() == self.information_dict['class_map']['NEGATIVE_CLASS']))[0]
        positive_D_d_mask = np.where((self.D_d.numpy() == 1) & (
            self.y.numpy() == self.information_dict['class_map']['POSITIVE_CLASS']))[0]
        
        # Sample a negative example from the advatanged class
        # and a positive example from the disadvantaged class
        neg_target_idx = np.random.choice(negative_D_a_mask, size=1)[0]
        pos_target_idx = np.random.choice(positive_D_d_mask, size=1)[0]
        
        return neg_target_idx, pos_target_idx

    def update_dataset(self, model):
        self.get_attack_directions(model)
        self.training_data = self.generate_poisoned_dataset()

    def generate_poisoned_dataset(self):
        return next(self.poisoned_generator)

    def poisoned_dataset_generator(self):
        """
        :return: The poisoned dataset.
        """
        # Sample two points from the dataset
        neg_target_idx, pos_target_idx = self.sample()

        self.x_target_neg, self.y_target_neg = self.X[neg_target_idx], self.y[neg_target_idx]
        self.x_target_pos, self.y_target_pos = self.X[pos_target_idx], self.y[pos_target_idx]
        
        # Generate the first version of the |εn| poissoned dataset
        n_pos_samples = int(np.sum(self.y.numpy() == self.information_dict['class_map']['NEGATIVE_CLASS']) * self.epsilon)
        n_neg_samples = int(np.sum(self.y.numpy() == self.information_dict['class_map']['POSITIVE_CLASS']) * self.epsilon)

        self.poisonedDataset = PoissonedDataset(
            X=torch.vstack([self.x_target_pos] * n_pos_samples + [self.x_target_neg] * n_neg_samples),
            Y=Tensor([self.y_target_pos] * n_pos_samples + [self.y_target_neg] * n_neg_samples).int()
        )

        whole_dataset = CustomConcatDataset(self.training_data, self.poisonedDataset)
        # poisoned_indices = torch.arange(len(self.poisonedDataset)) + len(self.training_data)
        # whole_dataset = self.project(whole_dataset, poisoned_indices)


        while True:
            # Train model with the clean and (current version of) poissoned dataset
            yield whole_dataset
            # Get the poisoned indices
            poisoned_indices = torch.arange(len(self.poisonedDataset)) + len(self.training_data)

            # get_attack_directions will be called in the training to update adversarial points

            # Update poisoned dataset
            self.poisonedDataset = PoissonedDataset(
                X=torch.vstack([self.x_target_pos] * n_pos_samples + [self.x_target_neg] * n_neg_samples),
                Y=Tensor([self.y_target_pos] * n_pos_samples + [self.y_target_neg] * n_neg_samples).int()
            )

            whole_dataset = CustomConcatDataset(self.training_data, self.poisonedDataset)
            whole_dataset = self.project(whole_dataset, poisoned_indices)

    def get_attack_directions(self, training_module):
        """
        :param training_module: The model to attack.
        :return: The adversarial directions.
        """
        self.x_target_neg = self.x_target_neg + self.eta * self.get_attack_direction(
                model=training_module.model,
                test_dataloader=self.val_dataloader(),  # TODO: check: I used validation set
                loss=lambda X, y: training_module.criterion(X, y) + self.lamda * self.fairness_loss(X, y),
                adverserial_point=(self.x_target_neg, self.y_target_neg),
            )
        self.x_target_pos = self.x_target_pos + self.eta *self.get_attack_direction(
                model=training_module.model,
                test_dataloader=self.val_dataloader(),  # TODO: check: I used validation set
                loss=lambda X, y: training_module.criterion(X, y) + self.lamda * self.fairness_loss(X, y),
                adverserial_point=(self.x_target_pos, self.y_target_pos),
            )
        # for x_adverserial in [[self.x_target_neg, self.y_target_neg], [self.x_target_pos, self.y_target_pos]]:  # TODO: check if inplace (should be)
        #     x_adverserial[0] = x_adverserial[0] + self.eta * self.get_attack_direction(
        #         model=training_module.model,
        #         test_dataloader=self.val_dataloader(),  # TODO: check: I used validation set
        #         loss=lambda X, y: training_module.criterion(X, y) + self.lamda * self.fairness_loss(X, y),
        #         adverserial_point=x_adverserial
        #     )

    @staticmethod
    def __flatten(tensors: Tuple[Tensor]) -> Tensor:
        """Concatenates a list of tensors of arbitrary shapes into a flat tensor.

        Args:
            tensors (Tuple[Tensor]): the tuple of tensors (for example a model's parameters)

        Returns:
            Tensor: a 1-D tensor containing all the parameters
        """
        return torch.cat([t.view(-1) for t in tensors])
    
    @staticmethod
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
    
    @staticmethod
    def __hvp(func: Callable, input: Union[Tensor, Tuple[Tensor]], v: Tensor) -> Tensor:
        """Computes the Hessian Vector Product of func at input with v.
        
        The Hessian Vector Product is the product between the Hessian of the function `func` at
        point `input` w.r.t. `input` and `v`, i.e. d^2 func(input) / d input^2 @ v.
        
        PyTorch has a built in function, :func:`torch.autograd.functional.hvp`, that effectively
        calculates this quantity, but it's limited to `v` matching the shape of `input`. Hence,
        this function acts as a wrapper for PyTorch's builtin function to handle the needs
        of :class:`IterativeAttackDatamodule`. What is more, we use the function :func:`torch.
        autograd.functional.vhp` instead of :func:`torch.autograd.functional.hvp`, which is
        computationally more effienct.

        Args:
            func (Callable): a Python Callable that acts as doubly differentiable function
            input (Union[Tensor, Tuple[Tensor]]): the point where the hessian of `func` is computed
            v (Tensor): a tensor whose primary dimension matches the number of elements in `input`

        Returns:
            Tensor: the Hessian Vector Product of func at input with v
        """
        hvp_columns = [] 
        
        # Iterate over columns of `v` with the same number of elements as `input`
        # TODO: check what happens if v has more than 2 dimensions (e.g. when we have image datasets)
        for v_column in v.T:
            # Reshape column to a tuple of tensors matching input
            v_column = InfluenceAttackDatamodule.__unflatten(v_column, input)
            
            # Calculate vhp for efficiency (since v is one-dimensional it's the same as hvp)
            _, hvp_column = vhp(func, input, v_column)
            
            # store hvp
            hvp_columns.append(InfluenceAttackDatamodule.__flatten(hvp_column))
        
        # vstack the hvp's to have shape as v (this is expected since hessian is a square matrix)
        return torch.vstack(hvp_columns).T

    @staticmethod
    def _g_theta(model: GenericModel, test_dataloader: DataLoader, loss: Callable) -> Tensor:
        """Computes the average gradients of the model parameters w.r.t the loss function on the
        test dataset.

        Args:
            model (GenericModel): a model deriving from the GenericModel class
            test_dataloader (DataLoader): the test dataloader
            loss (Callable): the loss function

        Returns:
            Tensor: the gradients of the model as a flattened tensor
        """
        model.zero_grad() # zero gradients for safety
        
        # Iterate over the test set and accumulate gradients (by not calling model.zero_grad())
        for X, y in test_dataloader:
            L = loss(model(X), y)
            L.backward()
        
        return model.get_grads() / len(test_dataloader)

    @staticmethod
    def _inverse_hvp(model: GenericModel, test_dataloader: DataLoader, loss: Callable, v: Tensor) -> Tensor:
        """Efficiently computes a numeric approximiation of the inverse Hessian Vector Product 
        between the test loss of a model w.r.t the model's parameters and a vector v.

        Args:
            model (GenericModel): a model deriving from the GenericModel class
            test_dataloader (DataLoader): the test dataloader
            loss (Callable): the loss function
            v (Tensor): a tensor

        Returns:
            Tensor: the inverse HVP estimate
        """
        inverse_hvp_estimate = v.clone().detach()   # first estimate of H^{-1}@v
        
        # Iterate over test batches
        for X, y in test_dataloader:
            def current_batch_loss(*theta):
                model.set_params(theta)
                return loss(model(X), y)
            
            # Iteratively update the estimate as H^{-1}@v <- v + (I - Hessian(L)) @ H^{-1}@v or
            # equivalently H^{-1}@v <- v + H^{-1}@v - hvp(L, H^{-1}@v), where L is the test loss
            inverse_hvp_estimate += v - InfluenceAttackDatamodule.__hvp(
                current_batch_loss,
                model.get_params(),
                inverse_hvp_estimate
            )
            
        return inverse_hvp_estimate

    @staticmethod
    def _loss_gradient_wrt_input_and_params(model: GenericModel, loss: Callable, point: Tuple[Tensor, Tensor]) -> Tensor:
        """Returns the second gradient of the loss function at point `point` w.r.t the input and model parameters.

        Args:
            model (GenericModel): a model deriving from the GenericModel class
            loss (Callable): the loss function
            point (Tuple[Tensor, Tensor]): a datapoint (X, y)

        Returns:
            Tensor: a 2-D tensor containing the second order gradients.
        """
        X, y = point
        X, y = X.unsqueeze(0), y.unsqueeze(0)   # create mini-batch of 1 sample to match loss expected shapes
        X.requires_grad_(True)                  # track gradients on input
        
        L = loss(model(X), y)                           # Loss
        L_first_grad = grad(L, X, create_graph=True)    # Gradient of loss w.r.t. input
        L_first_grad = L_first_grad[0].squeeze(0)       # Grad always returns a tuple, because it treats input as a tuple.
                                                        # In our case it treats X as (X, ), so we need to extact the first 
                                                        # element. Then we squeeze to discard the batch dimension
        
        # L_second_grad dimensions: num_params x num_input_features
        L_second_grad = torch.empty(InfluenceAttackDatamodule.__flatten(model.parameters()).shape + X.shape[1:])
        
        # Gradient requires scalar inputs. So to derive the second order derivative we need to use grad on every scalar 
        # in the first gradient (L_first_grad). `torch.autograd.functional.jacobian` is supposed to make this simpler, but
        # it is still in beta, and in our experiments it did not return correct results.
        for i, dL_dXi in enumerate(L_first_grad):
            L_second_grad[:, i] = InfluenceAttackDatamodule.__flatten(grad(dL_dXi, model.parameters(), create_graph=True))
            
        return L_second_grad

    @staticmethod
    def get_attack_direction(model: GenericModel, test_dataloader: DataLoader, loss: Callable, adverserial_point: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the direction of the influence attack as described in the papers:
         - Stronger data poisoning attacks break data sanitization defenses (https://arxiv.org/abs/1811.00741)
         - Exacerbating Algorithmic Bias through Fairness Attacks (https://arxiv.org/abs/2012.08723)

        Args:
            model (GenericModel): a model deriving from the GenericModel class
            test_dataloader (DataLoader): the test dataloader
            loss (Callable): the loss function
            adverserial_point (Tuple[Tensor, Tensor]): the adverserial datapoint (X, y)

        Returns:
            Tensor: a tensor with the direction of the influence attack
        """
        g_theta = InfluenceAttackDatamodule._g_theta(model, test_dataloader, loss)
        v = InfluenceAttackDatamodule._loss_gradient_wrt_input_and_params(model, loss, adverserial_point)
        inverse_hvp = InfluenceAttackDatamodule._inverse_hvp(model, test_dataloader, loss, v)
        
        return -g_theta @ inverse_hvp
