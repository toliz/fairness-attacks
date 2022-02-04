import abc
import unittest

import pytorch_lightning as pl
import torch
from scipy.spatial.distance import cdist as compute_distances
from torch import Tensor, IntTensor, BoolTensor
from torch.nn import BCEWithLogitsLoss

from attacks import anchoring, influence
from datamodules import GermanCreditDatamodule
from fairness import FairnessLoss
from trainingmodule import BinaryClassifier


class AttackTest(unittest.TestCase, metaclass=abc.ABCMeta):
    def setUp(self):
        """
        Set up the attack test by loading a Datamodule (German Credit) and storing the train dataset in memory.
        """
        self.dm = GermanCreditDatamodule('./data', 10)
        self.dataset = self.dm.get_train_dataset()

    @staticmethod
    def get_index(x: Tensor, X: Tensor) -> int:
        """
        Get the index of element `x` in tensor `X`.

        Args:
            x: the element to find
            X: the tensor containing all elements

        Returns: the element's index
        """
        return torch.all(X == x, dim=1).nonzero().item()

    def assert_classes(self, neg_idx: int, pos_idx: int):
        """
        Assert the classes of the two arguments, namely that the element at index `neg_idx` has a negative label,
        and the element at `pos_idx` has a positive one.

        Args:
            neg_idx: the index of the element with the negative label
            pos_idx: the index of the element with the positive label
        """
        self.assertEqual(self.dataset.Y[neg_idx], 0)
        self.assertEqual(self.dataset.Y[pos_idx], 1)

    def assert_demographics(self, adv_idx: int, disadv_idx: int):
        """
        Asserts the demographic groups of the two arguments, namely that the element at index `adv_idx` belongs to
        the advantaged class, and the element at `disadv_idx` to the disadvantaged.

        Args:
            adv_idx: the index of the element in the advantaged class
            disadv_idx: the index of the element in the disadvantaged class
        """
        self.assertEqual(self.dataset.adv_mask[adv_idx], 1)
        self.assertEqual(self.dataset.adv_mask[disadv_idx], 0)


class AnchoringTest(AttackTest):
    def assert_popular_percentile(self, x_popular: Tensor, mask: BoolTensor, percentile: float = 0.15):
        """
        Assert that the popular point selection algorithm based on the percentile rule is valid.

        Args:
            x_popular: the most popular point proposed by the implementated function
            mask: the binary mask to apply over the dataset to only keep the relevant points
            percentile: the percentile to use for the maximum distance comparison
        """
        # Extract the group of interest based on the mask
        group = self.dataset.X[mask]

        # Get the index of the proposed popular point
        popular_idx = self.get_index(x_popular, group)

        # Calculate the distances between all points in the masked data
        distances = torch.tensor(compute_distances(group, group, 'cityblock'))

        """ --- unimplemented approach ---
        # Calculate max_dist based on raw distances quantile
        triu_distances = distances.triu()
        nonzero_distances = triu_distances[triu_distances.nonzero(as_tuple=True)]
        max_dist = nonzero_distances.quantile(quantile).item()
        """

        # Calculate max_dist based on "mean point"
        mean_point = group.mean(dim=0).unsqueeze(dim=0)
        mean_dist = torch.tensor(compute_distances(mean_point, group, 'cityblock'))
        max_dist = mean_dist.quantile(percentile).item()

        # Find the index of the element with the most neighbors
        max_idx = (distances < max_dist).sum(dim=0).argmax().item()
        self.assertEqual(max_idx, popular_idx)

    def test_sampling_random(self):
        """
        Test the sampling method with random point selection.
        """
        # Get the negative+advantaged and positive+disadvantaged samples
        x_neg_adv, x_pos_disadv = anchoring._sample(self.dataset, 'random')

        # Get the indices of the two samples
        neg_adv_idx = self.get_index(x_neg_adv, self.dataset.X)
        pos_disadv_idx = self.get_index(x_pos_disadv, self.dataset.X)

        # Assert that we get the anticipated class for each sample
        self.assert_classes(neg_adv_idx, pos_disadv_idx)

        # Likewise for the demographic group
        self.assert_demographics(neg_adv_idx, pos_disadv_idx)

    def test_sampling_non_random_percentile(self):
        """
        Test the sampling method with non-random point selection, using the percentile distance rule for neighbors.
        """
        # Get the negative+advantaged and positive+disadvantaged samples
        x_neg_adv, x_pos_disadv = anchoring._sample(self.dataset, 'non-random', distances_type='percentile')

        # Get the indices of the two samples
        neg_adv_idx = self.get_index(x_neg_adv, self.dataset.X)
        pos_disadv_idx = self.get_index(x_pos_disadv, self.dataset.X)

        # Assert that we get the anticipated class for each sample
        self.assert_classes(neg_adv_idx, pos_disadv_idx)

        # Likewise for the demographic group
        self.assert_demographics(neg_adv_idx, pos_disadv_idx)

        # Also check if each sample has the most neighbors in its group
        neg_adv_mask = torch.logical_and(~self.dataset.Y.bool(), self.dataset.adv_mask)
        pos_disadv_mask = torch.logical_and(self.dataset.Y.bool(), ~self.dataset.adv_mask)

        # Assertions for type checking
        assert isinstance(neg_adv_mask, BoolTensor)
        assert isinstance(pos_disadv_mask, BoolTensor)

        # Assert that the proposed point is indeed popular, according to the percentile rule
        self.assert_popular_percentile(x_neg_adv, neg_adv_mask)
        self.assert_popular_percentile(x_pos_disadv, pos_disadv_mask)

    def test_perturb(self, tau: float = 0.1):
        """
        Test the perturbation method with the specified `tau`.

        Args:
            tau: the maximum allowed distance for the perturbed points
        """
        # Get the negative and positive samples
        x_target_neg, x_target_pos = anchoring._sample(self.dataset, 'random')

        # Generate the two sets, G_plus and G_minus
        for (x_target, generate_pos_adv) in ((x_target_neg, True), (x_target_pos, False)):
            G = anchoring._generate_perturbed_points(
                x_target=x_target,
                is_positive=generate_pos_adv,
                is_advantaged=generate_pos_adv,
                sensitive_idx=self.dm.get_sensitive_index(),
                tau=tau,
                n_perturbed=1
            )

            # Extract the features, label and advanced mask
            x, y, adv = G.sample()

            # Assert that we have the right demographic and label
            self.assertEqual(y, int(generate_pos_adv))
            self.assertEqual(adv, int(generate_pos_adv))

            # Assert constraint on distance (tau)
            dist = (x - x_target).norm().item()
            self.assertLessEqual(dist, tau)


class InfluenceTest(AttackTest):
    def setUp(self):
        """
        Set up the attack test by loading a Datamodule (German Credit) and storing the train & test datasets in memory.
        Define the performance and fairness losses as proposed by Mehrabi et al. (https://arxiv.org/abs/2012.08723)
        """
        # Call the parent class' setup method
        super().setUp()

        self.D_test = self.dm.get_test_dataset()

        self.perf_loss = BCEWithLogitsLoss()
        self.fair_loss = FairnessLoss(self.dm.get_sensitive_index())

        def adv_loss(model: BinaryClassifier, X: Tensor, Y: IntTensor, _: BoolTensor) -> Tensor:
            return self.perf_loss(model(X), Y.float()) + self.fair_loss(X, *model.get_params())

        self.adv_loss = adv_loss

    def test_sampling(self):
        """
        Test the sampling method, regarding the generated points' classes and demographics.
        """
        # Get the positive+advantaged and negative+disadvantaged samples
        x_pos_adv, x_neg_disadv = influence._sample(self.dataset)

        # Get the indices of the two samples
        pos_adv_idx = self.get_index(x_pos_adv, self.dataset.X)
        neg_disadv_idx = self.get_index(x_neg_disadv, self.dataset.X)

        # Assert that we get the anticipated class for each sample
        self.assert_classes(neg_disadv_idx, pos_adv_idx)

        # Likewise for the demographic group
        self.assert_demographics(pos_adv_idx, neg_disadv_idx)

    def generate_poisoned_dataset(self, eps: float = 0.1):
        """
        Helper function to generate the poisoned dataset, to be used with other test cases.

        Args:
            eps: the amount of poisoned points to generate, as a fraction of the training dataset's size
        """
        x_adv, y_adv = dict.fromkeys(['pos', 'neg']), dict.fromkeys(['pos', 'neg'])

        # Get the positive and negative samples
        x_adv['pos'], x_adv['neg'] = influence._sample(self.dataset)
        # Create their matching labels
        y_adv['pos'], y_adv['neg'] = torch.tensor(1, dtype=torch.int), torch.tensor(0, dtype=torch.int)

        # Calculate the amount of points to generate
        num_pos = int(eps * self.dataset.get_negative_count())
        num_neg = int(eps * self.dataset.get_positive_count())

        # Return the invocation of the build_dataset_from_points private method
        return influence._build_dataset_from_points(x_adv, y_adv, pos_copies=num_pos, neg_copies=num_neg)

    def test_build_poisoned_dataset(self):
        """
        Test the generation of the poisoned dataset.
        """
        # Generate the poisoned dataset with the implemented function
        D_p = self.generate_poisoned_dataset()

        # First element should be positive and advantaged
        self.assertEqual(D_p.Y[0], 1)
        self.assertEqual(D_p.adv_mask[0], 1)

        # Last element should be negative and disadvantaged
        self.assertEqual(D_p.Y[-1], 0)
        self.assertEqual(D_p.adv_mask[-1], 0)

    def test_g_theta(self):
        """
        Test the calculation of g_θ, regarding the shapes of the returned tensor.
        """
        # Create a binary classifier model to be used for the gradients calculation
        model = BinaryClassifier('LogisticRegression', self.dm.get_input_size())
        # Get g_θ with the implemented function
        g_theta = influence._compute_g_theta(model, self.adv_loss, self.D_test)

        # g_θ should have as many elements as the model's parameters, which is the
        # sum of the number of weights and the number of biases
        weights, biases = model.get_params()
        num_params = g_theta.shape[0]
        num_weights = weights.flatten().shape[0]
        num_biases = biases.flatten().shape[0]
        self.assertEqual(num_params, num_weights + num_biases)


if __name__ == '__main__':
    pl.seed_everything(123)
    unittest.main()
