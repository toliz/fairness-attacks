import pandas as pd
from attacks.datamodule import DataModule, CleanDataset
from abc import abstractmethod


class GenericAttackDataModule(DataModule):
    def __init__(self,
                 batch_size: int,
                 dataset: str,
                 path: str,
                 test_train_ratio: float = 0.2):
        super().__init__(batch_size=batch_size,
                         dataset=dataset,
                         path=path,
                         test_train_ratio=test_train_ratio)

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
            self.D_a = self.X[:, self.
                              information_dict['advantaged_column_index'] -
                              1] == self.information_dict['advantaged_label']
            self.D_d = self.X[:, self.
                              information_dict['advantaged_column_index'] -
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
