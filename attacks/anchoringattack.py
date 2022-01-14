import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from attacks.genericattack import GenericAttackDataModule

PATH = './data/'


class AnchoringAttackDatamodule(GenericAttackDataModule):
    def __init__(self, batch_size: int, dataset: str, path: str, method: str, epsilon: float, tau: float,
                 test_train_ratio: float = 0.2):
        """
        :param method: The method to use for anchoring.
        Options:
        - 'random' - Randomly select a point from the dataset.
        - 'non_random' - Select a popular point from the dataset.
        """
        super().__init__(batch_size=batch_size, dataset=dataset, path=path, test_train_ratio=test_train_ratio)
        self.method = method
        self.epsilon = epsilon
        self.tau = tau

    def attack(self):
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
        y_adv_neg = torch.zeros(len(x_adv_neg)) + self.information_dict['POSITIVE_CLASS']
        # Generate |D_c^{+}|ε| negative poisoned points (x_adv_pos, -1)
        # in the close vicinity of x_target_pos
        x_disadv_pos = self.attack_point(x_target_pos, advantaged=False)
        # Assign negative labels to the adversarial examples
        y_disadv_pos = torch.zeros(len(x_disadv_pos)) + self.information_dict['NEGATIVE_CLASS']
        # Return the adversarial examples
        x_adv_neg = torch.stack(x_adv_neg)
        x_disadv_pos = torch.stack(x_disadv_pos)
        return torch.cat([x_adv_neg, x_disadv_pos]), torch.cat([y_adv_neg, y_disadv_pos])

    def sample(self):
        # Sample a negative example from the advatanged class
        # and a positive example from the disadvantaged class
        np.random.seed(0)
        if self.method == 'random':
            # Randomly select a point from the dataset
            x_target_neg_idx = np.random.choice(
                np.where((self.D_a.numpy() == 1) & (self.y == self.information_dict['NEGATIVE_CLASS']))[0])
            x_target_pos_idx = np.random.choice(np.where((self.D_d.numpy() == 1) &
                                                         (self.y == self.information_dict['POSITIVE_CLASS']))[0])
        elif self.method == 'non_random':
            raise NotImplementedError(
                "Non-random anchoring is not implemented yet.")
        else:
            raise NotImplementedError("Unknown anchoring method.")
        return x_target_neg_idx, x_target_pos_idx

    def attack_point(self, x_target, advantaged: bool):
        """
        :param x_target: The point to attack.
        :return: The adversarial examples
        """
        # Get the adversarial examples
        x_adv = self.perturb(x_target, advantaged)
        return x_adv

    def perturb(self, x_target, advantaged: bool):
        """
        :param x_target: The point to attack.
        :return: The adversarial examples
        """
        # Calculate the number of points to perturb
        n_adv = int(self.epsilon * self.D_a.sum())
        n_disadv = int(self.epsilon * self.D_d.sum())
        if advantaged:
            N = n_adv
        else:
            N = n_disadv
        advantaged_or_not = "advantaged" if advantaged else "disadvantaged"
        print(f"Poisoning {N} {advantaged_or_not} points.")
        # Get the adversarial examples
        points = []
        mean = np.zeros_like(x_target)
        cov = 2 * np.eye(len(mean)) * self.tau ** 2
        while True:
            # Check if the adversarial example is distanced less
            # than tau from the target point
            # If not, perturb the adversarial example
            perturbation = np.random.multivariate_normal(mean, cov * 0.01, 1)[0, :]
            perturbation[self.information_dict['advantaged_column_index']] = 0
            x_adv = x_target + perturbation
            if not np.linalg.norm(x_adv - x_target) <= self.tau:
                pass
            else:
                points.append(x_adv)
                if len(points) == N:
                    break

        return points

    def project_to_feasible_set(self, x_adv, feasible_set):
        """
        :param x_adv: The adversarial examples.
        :return: The adversarial examples projected to the feasible set.
        """
        # Project the adversarial examples to the feasible set
        # Calculate the argmin of the distance between the adversarial
        # examples and the feasible set
        x_adv_feasible = []
        for x in x_adv:
            raise NotImplementedError(
                "Projection to feasible set is not implemented yet.")

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
        return PoissonedDataset(X_, y_)


class PoissonedDataset(Dataset):
    def __init__(self, X: Tensor, Y: Tensor):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    attack = AnchoringAttackDatamodule(1,
                                       dataset='German_Credit',
                                       path=PATH,
                                       test_train_ratio=0.2,
                                       method='random',
                                       epsilon=1,
                                       tau=0)
    # Attack the data
    attack.prepare_data()
    attack.setup()
    # Show the adversarial examples
    print(PoissonedDataset)
