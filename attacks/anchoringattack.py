import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from attacks.genericattack import GenericAttackDataModule
from attacks.datamodule import PoisonedDataset
from typing import List, Union, Tuple

PATH = './data/'


class AnchoringAttackDatamodule(GenericAttackDataModule):

    def __init__(
        self,
        batch_size: int,
        dataset: str,
        path: str,
        method: str,
        epsilon: float,
        tau: float,
        test_train_ratio: float = 0.2,
        projection_method: str = 'sphere',
        projection_radii: dict = None,
        alpha: float = 0.9,
    ) -> None:
        """
        :param method: The method to use for anchoring.
        Options:
        - 'random' - Randomly select a point from the dataset.
        - 'non_random' - Select a popular point from the dataset.
        """
        super().__init__(batch_size=batch_size,
                         dataset=dataset,
                         path=path,
                         test_train_ratio=test_train_ratio,
                         projection_method=projection_method,
                         projection_radii=projection_radii,
                         alpha=alpha,
                         epsilon=epsilon,)
        self.method = method
        self.tau = tau

    def attack(self) -> Tuple[Tensor, Tensor]:
        """
        :param method: The method to use for anchoring.
        Options:
        - 'random' - Randomly select a point from the dataset.
        - 'non_random' - Select a popular point from the dataset.
        :return: The adversarial examples.
        """

        # Sample a point from the dataset
        x_target_neg_idx, x_target_pos_idx = self.sample()
        # Get the point
        x_target_neg = self.X[x_target_neg_idx]
        x_target_pos = self.X[x_target_pos_idx]
        # Get the adversarial examples
        # Generate |D_c^{-}|ε| positive poisoned points (x_adv_neg, +1)
        # in the close vicinity of x_target_neg
        x_adv_neg = self.attack_point(x_target_neg, advantaged=True)
        # Assign positive labels to the adversarial examples
        y_adv_neg = torch.zeros(
            len(x_adv_neg)) + self.information_dict['class_map']['POSITIVE_CLASS']
        # Generate |D_c^{+}|ε| negative poisoned points (x_adv_pos, -1)
        # in the close vicinity of x_target_pos
        x_disadv_pos = self.attack_point(x_target_pos, advantaged=False)
        # Assign negative labels to the adversarial examples
        y_disadv_pos = torch.zeros(
            len(x_disadv_pos)) + self.information_dict['class_map']['NEGATIVE_CLASS']
        # Return the adversarial examples
        x_adv_neg = torch.stack(x_adv_neg)
        x_disadv_pos = torch.stack(x_disadv_pos)
        return torch.cat([x_adv_neg, x_disadv_pos]), torch.cat([y_adv_neg, y_disadv_pos])

    def sample(self) -> Tuple[int, int]:
        """
        :return: The indices of the points to attack.
        """
        # Sample a negative example from the advatanged class
        # and a positive example from the disadvantaged class
        negative_D_a_mask = np.where((self.D_a.numpy() == 1) & (
            self.y.numpy() == self.information_dict['class_map']['NEGATIVE_CLASS']))[0]
        positive_D_d_mask = np.where((self.D_d.numpy() == 1) & (
            self.y.numpy() == self.information_dict['class_map']['POSITIVE_CLASS']))[0]
        if self.method == 'random':
            np.random.seed(0)
            # Randomly select a point from the dataset
            x_target_neg_idx = np.random.choice(negative_D_a_mask, size=1)[0]
            x_target_pos_idx = np.random.choice(positive_D_d_mask, size=1)[0]
        elif self.method == 'non_random':
            """
            Non-random method: Select the most popular point from the dataset.
            The most popular point is the one that has the most neighbors with
            the same advantage class and label.
            """
            # Get the number of neighbors for each point
            neighbors_neg = self.get_neighbors(negative_D_a_mask)
            neighbors_pos = self.get_neighbors(positive_D_d_mask)
            # Get the most popular point
            x_target_neg_idx = np.argmax(neighbors_neg)
            x_target_pos_idx = np.argmax(neighbors_pos)
            # Get the most popular point

        else:
            raise NotImplementedError("Unknown anchoring method.")
        return x_target_neg_idx, x_target_pos_idx

    def get_neighbors(self,
                      mask: Union[np.ndarray,
                                  torch.Tensor,
                                  List[int]],
                      distance_threshold: float = 3,
                      distance_type: str = 'euclidean') -> np.ndarray:
        """
        :param mask: The mask of the points to consider.
        :param distance_threshold: The distance threshold to consider.
        :param distance_type: The distance type to consider.
        Options: 
        - 'euclidean'
        - 'manhattan'
        - TODO: More distance types
        :return: The number of neighbors for each point.
        """
        # Get the number of neighbors for each point in the masked dataset
        # Return the number of neighbors for each point in the original dataset
        neighbors = np.zeros(len(self.X))
        for idx in mask:
            # Get the neighbors
            neighbors[idx] = len(
                np.where(
                    self.get_distance(self.X[idx],
                                      self.X[mask],
                                      distance_type=distance_type) < distance_threshold)[0])
        return neighbors

    def get_distance(self,
                     x1: Union[np.ndarray,
                               torch.Tensor,
                               List[float]],
                     dataset: Union[np.ndarray,
                                    torch.Tensor],
                     distance_type: str = 'euclidean') -> Union[np.ndarray,
                                                                torch.Tensor]:
        """
        :param x1: The first point.
        :param dataset: The dataset to consider.
        :param distance_type: The distance type to consider.
        Options:
        - 'euclidean'
        - 'manhattan'
        return: The distance between x1 and each point in dataset.
        """
        if distance_type == 'euclidean':
            return np.linalg.norm(dataset - x1, axis=1)
        elif distance_type == 'manhattan':
            return np.sum(np.abs(dataset - x1), axis=1)
        else:
            raise NotImplementedError("Unknown distance type.")

    def attack_point(self,
                     x_target: Union[np.ndarray,
                                     torch.Tensor,
                                     List[float]],
                     advantaged: bool) -> List[np.ndarray]:
        """
        :param x_target: The point to attack.
        :return: The adversarial examples
        """
        # Get the adversarial examples
        x_adv = self.perturb(x_target, advantaged)
        return x_adv

    def perturb(self,
                x_target: Union[np.ndarray,
                                torch.Tensor,
                                List[float]],
                advantaged: bool) -> List[np.ndarray]:
        """
        :param x_target: The point to attack.
        :param advantaged: If True, the point is advantaged.
        :return: The adversarial examples
        """
        # Calculate the number of points to perturb
        n_adv = int(self.epsilon *
                    np.sum(self.y.numpy() == self.information_dict['class_map']['NEGATIVE_CLASS']))
        n_disadv = int(self.epsilon *
                       np.sum(self.y.numpy() == self.information_dict['class_map']['POSITIVE_CLASS']))
        if advantaged:
            N = n_adv
        else:
            N = n_disadv
        advantaged_or_not = "advantaged" if advantaged else "disadvantaged"
        print(f"Poisoning {N} {advantaged_or_not} points.")
        # Get the adversarial examples
        points = []
        mean = np.zeros_like(x_target)
        cov = 2 * np.eye(len(mean)) * self.tau**2
        while True:
            # Check if the adversarial example is distanced less
            # than tau from the target point
            # If not, perturb the adversarial example
            perturbation = np.random.multivariate_normal(mean, cov * 0.01, 1)[0, :]
            perturbation[self.information_dict['advantaged_column_index']] = 0  # TODO: Check if correct
            x_adv = x_target + perturbation
            if not np.linalg.norm(x_adv - x_target) <= self.tau:
                pass
            else:
                points.append(x_adv)
                if len(points) == N:
                    break

        return points

    def project_to_feasible_set(
        self,
        x_adv: Union[np.ndarray,
                     torch.Tensor,
                     List[float]],
    ):
        """
        :param x_adv: The adversarial examples.
        :param feasible_set: The feasible set.
        :return: The adversarial examples projected to the feasible set.
        """
        # Project the adversarial examples to the feasible set
        # Calculate the argmin of the distance between the adversarial
        # examples and the feasible set

    def generate_poisoned_dataset(self):
        """
        :return: The poisoned dataset.
        """
        poisoned_X, poisoned_y = self.attack()
        poisoned_X = poisoned_X.float()
        poisoned_y = poisoned_y.int()
        # Shuffle the poisoned dataset
        permutation = np.random.permutation(len(poisoned_X))
        poisoned_X = poisoned_X[permutation]
        poisoned_y = poisoned_y[permutation]
        # Append to original dataset
        X_ = torch.cat((self.X, poisoned_X)).float()
        y_ = torch.cat((torch.tensor(self.y), poisoned_y))
        poisoned_indices = torch.arange(len(poisoned_X)) + len(self.X)
        self.PoisonedDataset = PoisonedDataset(X_, y_)
        self.PoisonedDataset = self.project(self.PoisonedDataset, poisoned_indices)
        return self.PoisonedDataset
