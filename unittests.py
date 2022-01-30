import abc
import unittest

import pytorch_lightning as pl
import torch

from scipy.spatial.distance import cdist as compute_distances
from torch.nn import BCEWithLogitsLoss

from attacks import anchoring, influence
from datamodules import GermanCreditDatamodule
from fairness import FairnessLoss
from trainingmodule import BinaryClassifier


class AttackTest(unittest.TestCase, metaclass=abc.ABCMeta):
    def setUp(self):
        self.dm = GermanCreditDatamodule('./data', 10)
        self.dataset = self.dm.get_train_dataset()

    @staticmethod
    def get_index(x: torch.Tensor, X: torch.Tensor) -> int:
        idx = torch.all(X == x, dim=1).nonzero().item()
        return idx

    def assert_classes(self, neg_idx: int, pos_idx: int):
        self.assertEqual(self.dataset.Y[neg_idx], 0)
        self.assertEqual(self.dataset.Y[pos_idx], 1)

    def assert_demographics(self, adv_idx: int, disadv_idx: int):
        self.assertEqual(self.dataset.adv_mask[adv_idx], 1)
        self.assertEqual(self.dataset.adv_mask[disadv_idx], 0)

class AnchoringTest(AttackTest):
    def assert_popular(self, x_popular: torch.Tensor, mask: torch.BoolTensor, quantile: float = 0.15):
        group = self.dataset.X[mask]
        popular_idx = self.get_index(x_popular, group)

        distances = torch.tensor(compute_distances(group, group, 'cityblock'))

        # # calculate max_dist based on raw distances quantile
        # triu_distances = distances.triu()
        # nonzero_distances = triu_distances[triu_distances.nonzero(as_tuple=True)]
        # max_dist = nonzero_distances.quantile(quantile).item()

        # calculate max_dist based on "mean point"
        mean_point = group.mean(dim=0).unsqueeze(dim=0)
        mean_dist = torch.tensor(compute_distances(mean_point, group, 'cityblock'))
        max_dist = mean_dist.quantile(quantile).item()

        max_idx = (distances < max_dist).sum(dim=0).argmax().item()
        self.assertEqual(max_idx, popular_idx)

    def test_sampling_random(self):
        x_neg_adv, x_pos_disadv = anchoring._sample(self.dataset, 'random')
        neg_adv_idx = self.get_index(x_neg_adv, self.dataset.X)
        pos_disadv_idx = self.get_index(x_pos_disadv, self.dataset.X)

        # assert that we get the anticipated class for each sample
        self.assert_classes(neg_adv_idx, pos_disadv_idx)

        # likewise for the demographic group
        self.assert_demographics(neg_adv_idx, pos_disadv_idx)

    def test_sampling_non_random(self):
        x_neg_adv, x_pos_disadv = anchoring._sample(self.dataset, 'non-random', distances_type='percentile')
        neg_adv_idx = self.get_index(x_neg_adv, self.dataset.X)
        pos_disadv_idx = self.get_index(x_pos_disadv, self.dataset.X)

        # assert that we get the anticipated class for each sample
        self.assert_classes(neg_adv_idx, pos_disadv_idx)

        # likewise for the demographic group
        self.assert_demographics(neg_adv_idx, pos_disadv_idx)

        # also check if each sample has the most neighbors in its group
        neg_adv_mask = torch.logical_and(~self.dataset.Y.bool(), self.dataset.adv_mask)
        pos_disadv_mask = torch.logical_and(self.dataset.Y.bool(), ~self.dataset.adv_mask)

        assert isinstance(neg_adv_mask, torch.BoolTensor)
        assert isinstance(pos_disadv_mask, torch.BoolTensor)
        self.assert_popular(x_neg_adv, neg_adv_mask)
        self.assert_popular(x_pos_disadv, pos_disadv_mask)

    def test_perturb(self):
        tau = 0.1
        x_target_neg, x_target_pos = anchoring._sample(self.dataset, 'random')

        for (x_target, generate_pos_adv) in ((x_target_neg, True), (x_target_pos, False)):
            G = anchoring._generate_perturbed_points(
                x_target=x_target,
                is_positive=generate_pos_adv,
                is_advantaged=generate_pos_adv,
                sensitive_idx=self.dm.get_sensitive_index(),
                tau=tau,
                n_perturbed=1
            )

            x, y, adv = G.sample()

            # assert that we have the right demographic and label
            self.assertEqual(y, int(generate_pos_adv))
            self.assertEqual(adv, int(generate_pos_adv))

            # assert constraint on distance (tau)
            dist = (x - x_target).norm().item()
            self.assertLessEqual(dist, tau)

class InfluenceTest(AttackTest):
    def setUp(self):
        super().setUp()

        self.D_test = self.dm.get_test_dataset()

        self.perf_loss = BCEWithLogitsLoss()
        self.fair_loss = FairnessLoss(self.dm.get_sensitive_index())

        def adv_loss(model: BinaryClassifier, X: torch.Tensor, Y: torch.IntTensor, _: torch.BoolTensor) -> torch.Tensor:
            return self.perf_loss(model(X), Y.float()) + self.fair_loss(X, *model.get_params())

        self.adv_loss = adv_loss

    def test_sampling(self):
        x_pos_adv, x_neg_disadv = influence._sample(self.dataset)
        pos_adv_idx = self.get_index(x_pos_adv, self.dataset.X)
        neg_disadv_idx = self.get_index(x_neg_disadv, self.dataset.X)

        # assert that we get the anticipated class for each sample
        self.assert_classes(neg_disadv_idx, pos_adv_idx)

        # likewise for the demographic group
        self.assert_demographics(pos_adv_idx, neg_disadv_idx)

    def generate_poisoned_dataset(self):
        x_adv, y_adv = dict.fromkeys(['pos', 'neg']), dict.fromkeys(['pos', 'neg'])

        x_adv['pos'], x_adv['neg'] = influence._sample(self.dataset)
        y_adv['pos'], y_adv['neg'] = 1, 0

        num_pos = int(0.1 * self.dataset.get_negative_count())
        num_neg = int(0.1 * self.dataset.get_positive_count())
        return influence._build_dataset_from_points(x_adv, y_adv, pos_copies=num_pos, neg_copies=num_neg)

    def test_build_poisoned_dataset(self):
        D_p = self.generate_poisoned_dataset()

        # first element should be positive and advantaged
        self.assertEqual(D_p.Y[0], 1)
        self.assertEqual(D_p.adv_mask[0], 1)

        # last element should be negative and disadvantaged
        self.assertEqual(D_p.Y[-1], 0)
        self.assertEqual(D_p.adv_mask[-1], 0)

    def test_g_theta(self):
        model = BinaryClassifier('LogisticRegression', self.dm.get_input_size())
        g_theta = influence._compute_g_theta(model, self.D_test, self.adv_loss)

        # g_theta should have as many elements as the model's parameters,
        # which is the sum of the weights and the biases
        weights, biases = model.get_params()
        num_params = g_theta.shape[0]
        num_weights = weights.flatten().shape[0]
        num_biases = biases.flatten().shape[0]
        self.assertEqual(num_params, num_weights + num_biases)

if __name__ == '__main__':
    pl.seed_everything(123)
    unittest.main()
