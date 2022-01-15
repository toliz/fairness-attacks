import numpy as np
import torch
import pandas as pd
from attacks.datamodule import DataModule, CleanDataset, PoissonedDataset
from abc import abstractmethod
import numpy


class GenericAttackDataModule(DataModule):

    def __init__(
        self,
        batch_size: int,
        dataset: str,
        path: str,
        test_train_ratio: float = 0.2,
        projection_method: str = 'sphere',
        projection_radii: dict = None,
        alpha: float = 0.9,
    ):
        super().__init__(batch_size=batch_size,
                         dataset=dataset,
                         path=path,
                         test_train_ratio=test_train_ratio)
        """
        Initialize the GenericAttackDataModule
        :param batch_size: the batch size
        :param dataset: the dataset to use
        :param path: the path to the dataset
        :param test_train_ratio: the ratio of the test data to the training data
        :param projection_method: the method to use for projection
        :param projection_radii: the radii to use for projection
        :param alpha: Fraction of the maximum radius per class 
        to use for the projection.
        """

        self.projection_method = projection_method
        self.projection_radii = projection_radii
        self.alpha = alpha

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

            # set up for the attack
            self.X = self.training_data[:][0]
            self.y = self.training_data[:][1]
            self.D_a = self.X[:,
                              self.information_dict['advantaged_column_index'] -
                              1] == self.information_dict['advantaged_label']
            self.D_d = self.X[:,
                              self.information_dict['advantaged_column_index'] -
                              1] != self.information_dict['advantaged_label']

            # attack the training data
            self.training_data = self.generate_poisoned_dataset()

            self.val_data = CleanDataset(self.val_data)

        # Set the test dataset
        if stage == 'test' or stage is None:
            self.test_data = CleanDataset(self.test_data)

    @abstractmethod
    def generate_poisoned_dataset(self):
        pass

    def get_centroids(self) -> numpy.ndarray:
        """
        Returns the centroids of the training data
        """
        classes = list(self.information_dict['class_map'].values())
        num_features = self.X.shape[1]
        centroids = numpy.zeros((len(classes), num_features))
        for c in classes:
            centroids[c] = numpy.mean(self.X[self.y == c].detach().cpu().numpy(), axis=0)
        return centroids

    def get_max_radii_from_centroids(self) -> dict:
        """
        Returns the maximum distance from the centroids of the training data
        """
        centroids = self.get_centroids()
        max_radii = {}
        for c in self.information_dict['class_map'].values():
            max_radii[c] = numpy.max(numpy.linalg.norm(centroids[c] - self.X[self.y == c].detach().cpu().numpy(), axis=1))
        return max_radii

    def get_class_counts(self) -> numpy.ndarray:
        """
        Returns the counts of the classes in the training data
        """
        classes = self.information_dict['class_map'].values()
        counts = numpy.zeros(len(classes))
        for i, c in enumerate(classes):
            counts[i] = numpy.sum(self.y == c)
        return counts

    def get_class_probabilities(self) -> numpy.ndarray:
        """
        Returns the probability of each class in the training data
        """
        counts = self.get_class_counts()
        return counts / numpy.sum(counts)

    def project(self, dataset: PoissonedDataset) -> PoissonedDataset:
        """
        Project the dataset
        :param dataset: the dataset to project 
        :return: a new dataset with anomalous points projected onto
        a feasible set
        """
        # If projection radii are not specified, use the maximum radii
        # times alpha
        if not self.projection_radii:
            radii = self.get_max_radii_from_centroids()
            for c in radii:
                radii[c] *= self.alpha
        else:
            radii = self.projection_radii

        if self.projection_method == 'sphere':
            return self.project_onto_sphere(dataset=dataset, radii=radii)
        elif self.projection_method == 'slab':
            return self.project_onto_slab(dataset=dataset, radii=radii)
        else:
            raise NotImplementedError(
                f'Projection method {self.projection_method} is not implemented')

    @staticmethod
    def project_onto_sphere(dataset: PoissonedDataset, radii: dict) -> PoissonedDataset:
        """Project onto sphere method
        
        :dataset: the dataset with the poissoned data
        :radii: a dictionary with the desired radius for each class
        s
        :return: a new dataset with anomalous points projected onto slab
        """
        X, Y = dataset.X.detach().clone(), dataset.Y.detach().clone()
        classes = set(list(Y.cpu().numpy()))

        for c in classes:
            # Iterate over classes and get the center and desired radius for each class
            center = X[Y == c].mean()
            radius = radii[c]

            # Finds datatoints shifts and distances from their center
            shifts_from_center = X[Y == c] - center
            dists_from_center = np.linalg.norm(shifts_from_center, axis=1)

            # Spot anomalous datapoints
            anomalous_indices = dists_from_center > radius

            # Project anomalous datapoints on a sphere with the desired radius
            print(type(dists_from_center[anomalous_indices]))
            shifts_from_center[
                anomalous_indices] *= radius / dists_from_center[anomalous_indices].reshape(-1,
                                                                                         1)
            X[Y == c] = shifts_from_center + center

        return PoissonedDataset(X, Y)

    @staticmethod
    def project_onto_slab(dataset: PoissonedDataset, radii: dict) -> PoissonedDataset:
        """
        Project onto slab method - as defined in the paper "Certified Defenses for Data Poisoning
        Attacks" (https://arxiv.org/abs/1706.03691)
        
        :dataset: the dataset with the poissoned data
        :radii: a dictionary with the desired radius for each class
        
        :return: a new dataset with anomalous points projected onto slab
        """
        X, Y = dataset.X.detach().clone(), dataset.Y.detach().clone()
        classes = set(list(Y.cpu().numpy()))

        # Assert the given dataset has binary labels
        assert len(classes) == 2

        # Find the vector connecting the centers of the two clusters
        v = X[Y == classes[1]].mean() - X[Y == classes[0]].mean()

        for c in classes:
            # Iterate over classes and get the center and desired radius for each class
            center = X[Y == c].mean()
            radius = radii[c]

            # Finds shifts and distances along the v vector
            dists_along_v = torch.dot(X[Y == c] - center, v.T)
            shifts_along_v = (dists_along_v - torch.clip(dists_along_v,
                                                         -radius,
                                                         radius)).view(1,
                                                                       -1)

            X[Y == c] -= torch.dot(shifts_along_v.T, v)

        return PoissonedDataset(X, Y)
