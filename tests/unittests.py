import unittest
import os
import pandas as pd
from ..attacks.datamodule import DataModule, CleanDataset
from ..attacks.genericattack import GenericAttackDataModule
from ..attacks.anchoringattack import AnchoringAttackDataModule

def test_anchoring_attack():
    # Test the Anchoring Attack
    # Set up the data
    dataset = 'GermanCredit'
    path = os.path.join('..', 'data', 'GermanCredit.csv')
    test_train_ratio = 0.2
    data_module = AnchoringAttackDataModule(1, dataset, path, test_train_ratio)
    # Set up the data module
    data_module.setup()
    # Test the data module
    unittest.TestCase.assertEqual(data_module.batch_size, 1)
    unittest.TestCase.assertEqual(data_module.dataset, 'GermanCredit')
    unittest.TestCase.assertEqual(data_module.path, path)
    unittest.TestCase.assertEqual(data_module.test_train_ratio, test_train_ratio)

    # Test the training data
    unittest.TestCase.assertEqual(len(data_module.training_data), len(data_module.training_data.loc[:, 'Features']))
    unittest.TestCase.assertEqual(len(data_module.training_data), len(data_module.training_data.loc[:, 'Class']))
    unittest.TestCase.assertEqual(len(data_module.training_data), len(data_module.training_data.loc[:, 'Advantage']))

    # Test the validation data
    unittest.TestCase.assertEqual(len(data_module.val_data), len(data_module.val_data.loc[:, 'Features']))
    unittest.TestCase.assertEqual(len(data_module.val_data), len(data_module.val_data.loc[:, 'Class']))
    unittest.TestCase.assertEqual(len(data_module.val_data), len(data_module.val_data.loc[:, 'Advantage']))

    # Test the test data
    unittest.TestCase.assertEqual(len(data_module.test_data), len(data_module.test_data.loc[:, 'Features']))
    unittest.TestCase.assertEqual(len(data_module.test_data), len(data_module.test_data.loc[:, 'Class']))
    unittest.TestCase.assertEqual(len(data_module.test_data), len(data_module.test_data.loc[:, 'Advantage']))

    # Test the training dataloader
    unittest.TestCase.assertEqual(len(data_module.train_dataloader()), len(data_module.training_data))

    # Test the validation dataloader
    unittest.TestCase.assertEqual(len(data_module.val_dataloader()), len(data_module.val_data))

    # Test the test dataloader
    unittest.TestCase.assertEqual(len(data_module.test_dataloader()), len(data_module.test_data))

    # Test the generic attack data module
    generic_data_module = GenericAttackDataModule(1, dataset, path, test_train_ratio)
    generic_data_module.setup()
    unittest.TestCase.assertEqual(len(generic_data_module.train_dataloader()), len(generic_data_module.training_data))
    unittest.TestCase.assertEqual(len(generic_data_module.val_dataloader()), len(generic_data_module.val_data))
    unittest.TestCase.assertEqual(len(generic_data_module.test_dataloader()), len(generic_data_module.test_data))

    # Test the clean dataset
    clean_dataset = CleanDataset(dataset, path)
    clean_dataset.setup()
    unittest.TestCase.assertEqual(len(clean_dataset.data), len(clean_dataset.data.loc[:, 'Features']))
    unittest.TestCase.assertEqual(len(clean_dataset.data), len(clean_dataset.data.loc[:, 'Class']))
    unittest.TestCase.assertEqual(len(clean_dataset.data), len(clean_dataset.data.loc[:, 'Advantage']))
    unittest.TestCase.assertEqual(len(clean_dataset.dataloader()), len(clean_dataset.data))

if __name__ == '__main__':
    test_anchoring_attack()