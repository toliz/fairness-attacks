from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelBinarizer
import torch
import numpy as np
from pandas import DataFrame


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, dataset: str, path: str, test_train_ratio: float = 0.2):
        super().__init__()

        self.batch_size = batch_size
        self.dataset = dataset
        self.path = path
        self.test_train_ratio = test_train_ratio
        self.num_classes = 2
        self.information_dict = {}

    def prepare_data(self):
        # Download data if not found in the path
        if self.dataset == 'German_Credit':
            if not os.path.isfile(self.path + 'German_Credit.csv'):
                # Load data from link
                df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                                 names=['Attribute' + str(i) for i in range(1, 21)] + ['Class'], delim_whitespace=True)
                # Get the Class into the right form
                df.loc[:, 'Class'] = df.loc[:, 'Class'] - 1
                # Find if the datapoint has advantage. TRUE if he/she is from Germany
                df['Advantage'] = df['Attribute20'] == 'A202'
                self.information_dict['advantaged_indices'] = df[df['Advantage'] == True].index
                # Save the data to memory
                df.to_csv(self.path + 'German_Credit.csv', index=False)
                print(self.dataset + ' Dataset Downloaded!')

        if self.dataset == 'Drug_Consumption':
            if not os.path.isfile(self.path + 'Drug_Consumption.csv'):
                # Load data from link
                df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data'
                                 , names=['Attribute' + str(i) for i in range(1, 33)])
                # Get the Class. 1 if he/she has used cocaine
                df['Class'] = (df['Attribute21'] != 'CL0').astype(int)
                # Find if the datapoint has advantage. TRUE if woman
                df['Advantage'] = df['Attribute3'] == 0.48246
                # Drop redundant columns
                df = df.drop(columns=['Attribute' + str(i) for i in range(14, 33)])
                # Save data to memory
                df.to_csv(self.path + 'Drug_Consumption.csv', index=False)
                print(self.dataset + ' Dataset Downloaded!')

    def get_input_size(self):
        return self.input_size

    def get_num_classes(self):
        return self.num_classes

    def split_data(self, df: DataFrame, test_size: float, shuffle: bool):
        # Split the DataFrame and reset index
        df_train, df_test = train_test_split(df, test_size=test_size, shuffle=shuffle)
        df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
        return df_train, df_test

    def process_data(self):
        if self.dataset == 'German_Credit':
            self.process_German_Credit_dataset()
        if self.dataset == 'Drug_Consumption':
            self.process_Drug_Consumption_dataset()

    def process_German_Credit_dataset(self):
        # Get the id of the numerical and qualitative attributes
        numerical_attributes = [2, 5, 8, 11, 13, 16, 18]
        qualitative_attributes = [1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20]
        self.information_dict['numerical_attributes'] = numerical_attributes
        self.information_dict['qualitative_attributes'] = qualitative_attributes
        # To be used for attack
        self.information_dict['advantaged_column_index'] = 20
        self.information_dict['advantaged_class'] = 'A202'
        self.information_dict['POSITIVE_CLASS'] = 0
        self.information_dict['NEGATIVE_CLASS'] = 1
        # One-Hot encoding for qualitative attributes
        for i in qualitative_attributes:
            attribute = 'Attribute' + str(i)
            enc = LabelBinarizer()
            enc.fit(self.training_data[attribute].values)
            self.training_data.loc[:, attribute] = self.training_data.loc[:, attribute].apply(
                lambda x: enc.transform([x])[0].astype(float))
            self.test_data.loc[:, attribute] = self.test_data.loc[:, attribute].apply(
                lambda x: enc.transform([x])[0].astype(float))

            if i == self.information_dict['advantaged_column_index']:
                self.information_dict['advantaged_label'] = enc.transform([self.information_dict['advantaged_class']])[0][0]

        # Minmax normalization for numerical attributes
        for i in numerical_attributes:
            attribute = 'Attribute' + str(i)
            max = self.training_data.loc[:, attribute].max()
            min = self.training_data.loc[:, attribute].min()
            self.training_data.loc[:, attribute] = self.training_data.loc[:, attribute].apply(
                lambda x: [(x - min) / (max - min)])
            self.test_data.loc[:, attribute] = self.test_data.loc[:, attribute].apply(
                lambda x: [np.clip((x - min) / (max - min), 0., 1.)])

        # Combine all attributes to one column
        self.create_column_with_features()

        # Get the input size needed for the model
        self.set_input_size()

    def process_Drug_Consumption_dataset(self):
        # Normalize IDs
        mean = self.training_data.loc[:, 'Attribute1'].mean()
        std = self.training_data.loc[:, 'Attribute1'].std()
        self.training_data.loc[:, 'Attribute1'] = (self.training_data.loc[:, 'Attribute1'] - mean)/std
        self.test_data.loc[:, 'Attribute1'] = (self.test_data.loc[:, 'Attribute1'] - mean)/std

        # Combine all attributes to one column
        self.create_column_with_features(fucn=np.hstack)

        # Get the input size needed for the model
        self.set_input_size()

    def create_column_with_features(self, non_attr_columns: list = ['Class', 'Advantage'], fucn: callable = np.concatenate):
        # Combine all attribute columns to one column containing one flat array
        self.training_data.loc[:, 'Features'] = self.training_data.loc[:, ~self.training_data.columns.isin(non_attr_columns)].apply(fucn, axis=1)
        self.test_data.loc[:, 'Features'] = self.test_data.loc[:, ~self.test_data.columns.isin(non_attr_columns)].apply(fucn, axis=1)

    def set_input_size(self):
        # Set the input size based on the length of the feature array
        self.input_size = len(self.training_data.loc[0, 'Features'])

    def setup(self, stage=None):
        df = pd.read_csv(self.path + self.dataset + '.csv')

        # Split and process the data
        self.training_data, self.test_data = self.split_data(df, test_size=self.test_train_ratio, shuffle=True)
        self.process_data()

        # Set the training and validation dataset
        if stage == 'fit' or stage is None:
            self.training_data, self.val_data = self.split_data(self.training_data,
                                                                test_size=self.test_train_ratio, shuffle=True)
            self.training_data = CleanDataset(self.training_data)
            self.val_data = CleanDataset(self.val_data)

        # Set the test dataset
        if stage == 'test' or stage is None:
            self.test_data = CleanDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class CleanDataset(Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        features = torch.tensor(self.dataset.loc[self.dataset.index[index], 'Features']).float()
        labels = self.dataset.loc[self.dataset.index[index], 'Class']
        return features, labels

    def __len__(self):
        return len(self.dataset)

    def get_advantaged_points(self):
        advantaged_points = self.dataset[self.dataset['Advantage']]
        features = torch.tensor(advantaged_points.loc[:, 'Features']).float()
        labels = advantaged_points.loc[:, 'Class']
        return features, labels