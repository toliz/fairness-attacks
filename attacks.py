import datamodule
import os
import pandas as pd
import numpy as np
import torch

PATH = 'data/'


class GenericAttack(datamodule.DataModule):
    def __init__(self,
                 dataset: str,
                 path: str,
                 test_train_ratio: float = 0.2) -> None:
        super().__init__(1, dataset, path, test_train_ratio)
        self.prepare_data()
        self.setup()
        self.X = self.training_data[:][0] # ??? How to find X
        self.y = self.training_data[:][1] # ??? How to find y
        # How to find the advantaged_index?
        self.D_a = self.X[self.advantaged_indices]
        self.D_d = self.X[~self.advantaged_indices]

    def setup(self):
        df = pd.read_csv(self.path + self.dataset + '.csv')
        # Split and process the data
        self.training_data, self.test_data = self.split_data(
            df, test_size=self.test_train_ratio, shuffle=True)
        self.process_data()


class AnchoringAttack(GenericAttack):
    def __init__(self, dataset: str, path: str, test_train_ratio: str,
                 method: str, epsilon: float, tau: float) -> None:
        """
        :param method: The method to use for anchoring.
        Options:
        - 'random' - Randomly select a point from the dataset.
        - 'non_random' - Select a popular point from the dataset.
        """
        super().__init__(dataset, path, test_train_ratio)
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
        x_target_neg_idx, x_target_pos_idx = self.sample(self.method)
        # Get the point
        x_target_neg = self.X[x_target_neg_idx]
        x_target_pos = self.X[x_target_pos_idx]
        # Get the labels
        y_target_neg = self.y[x_target_neg_idx]
        y_target_pos = self.y[x_target_pos_idx]
        # Get the adversarial examples
        x_adv_neg = self.attack_point(x_target_neg)
        x_adv_pos = self.attack_point(x_target_pos)
        # Return the adversarial examples
        return x_adv_neg, x_adv_pos

    def sample(self):
        # Sample a negative example from the advatanged class
        # and a positive example from the disadvantaged class
        np.random.seed(0)
        if self.method == 'random':
            # Randomly select a point from the dataset
            x_target_neg_idx = np.random.choice(
                np.intersect1d(self.D_a,
                               np.where(self.y == 2)[0]))
            x_target_pos_idx = np.random.choice(
                np.intersect1d(self.D_d,
                               np.where(self.y == 1)[0]))
            return x_target_neg_idx, x_target_pos_idx
        elif self.method == 'non_random':
            raise NotImplementedError(
                "Non-random anchoring is not implemented yet.")
        else:
            raise NotImplementedError("Unknown anchoring method.")

    def attack_point(self, x_target):
        """
        :param x_target: The point to attack.
        :return: The adversarial examples
        """
        # Get the adversarial examples
        x_adv = self.perturb(x_target, self.epsilon, self.tau)
        return x_adv

    def perturb(self, x_target):
        """
        :param x_target: The point to attack.
        :return: The adversarial examples
        """
        # Calculate the number of points to perturb
        n_adv = int(self.epsilon * len(self.D_a))
        # Get the adversarial examples
        points = []
        x_adv = x_target + torch.randn_like(x_target)
        while True:
            # Check if the adversarial example is distanced less
            # than tau from the target point
            if np.linalg.norm(x_adv - x_target) < self.tau:
                break
            else:
                # If not, perturb the adversarial example
                x_adv = x_target + torch.randn_like(x_target)
                points.append(x_adv)
                if len(points) == n_adv:
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


# Test the attack
if __name__ == '__main__':
    attack = AnchoringAttack(dataset='German_Credit',
                             path=PATH,
                             test_train_ratio=0.2,
                             method='random',
                             epsilon=0.1,
                             tau=1)
    # Attack the data
    x_adv_neg, x_adv_pos = attack.attack(method='random')
    # Show the adversarial examples
    print(x_adv_neg)
    print(x_adv_pos)
