#!/usr/bin/env python
# coding: utf-8

# First, let's import all necessary libraries:

# In[12]:


import collections
import logging
import numpy as np
import pickle
import pytorch_lightning as pl
import torch

from copy import deepcopy
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from attacks import influence_attack, anchoring_attack
from datamodules import GermanCreditDatamodule, CompasDatamodule, DrugConsumptionDatamodule
from fairness import FairnessLoss
from trainingmodule import BinaryClassifier

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# Now we create a **general attack function** that handles all different attack methods

# In[13]:


def attack(dm, model, eps, method):
    if method in ['IAF', 'Koh et al', 'Solans et al']:
        bce_loss, fairness_loss = BCEWithLogitsLoss(), FairnessLoss(dm.get_sensitive_index())
        
        if method == 'IAF':
            # Create adversarial loss according to Mehrabi et al.
            adv_loss = lambda _model, X, y, adv_mask: (
                    bce_loss(_model(X), y.float()) + 1.0 * fairness_loss(X, *_model.get_params())
            )
        elif method == 'Koh et al':
            # Create adversarial loss according to Koh et al.
            adv_loss = lambda _model, X, y, adv_mask: bce_loss(_model(X), y.float())
        else:
            # Create adversarial loss according to Solans et al.
            adv_loss = lambda _model, X, y, adv_mask: (
                bce_loss(_model(X[~adv_mask]), y[~adv_mask].float()) + \
                bce_loss(_model(X[adv_mask]), y[adv_mask].float()) * torch.sum(~adv_mask) / torch.sum(adv_mask)
            )
        
        # Create new training pipeline to use in influence attack
        trainer = pl.Trainer(
            max_epochs=100,
            gpus=1 if torch.cuda.is_available() else 0,
            enable_model_summary=False,
            enable_progress_bar=False,
            log_every_n_steps=1,
            callbacks=[EarlyStopping(monitor="train_acc", mode="max", patience=10)]
        )

        poisoned_dataset = influence_attack(
            model=model,
            datamodule=dm,
            trainer=trainer,
            adv_loss=adv_loss,
            eps=eps,
            eta=0.01,
            attack_iters=100,
        )
    elif method in ['RAA', 'NRAA']:
        poisoned_dataset = anchoring_attack(
            D_c=dm.get_train_dataset(),
            sensitive_idx=dm.get_sensitive_index(),
            eps=eps,
            tau=0,
            sampling_method='random' if method == 'RAA' else 'non-random',
        )
    else:
        raise ValueError(f'Unknown attack {method}.')
    
    # Create deep copy of the original dataset and poison the copy
    dm = deepcopy(dm)
    dm.update_train_dataset(poisoned_dataset)

    return dm


# and a **nested dictionary**, which is convinient to store results for multiple datasets and metrics:

# In[14]:


def nested_dict():
   return collections.defaultdict(nested_dict)


# Finally, iterate over all possible combination of Figure 2 in Mehrabi et al.

# In[15]:


from pathlib import Path
if Path('/home/lcur0488/different_epsilons.pkl').is_file():
    with open('/home/lcur0488/different_epsilons.pkl', 'rb') as f:
	    results = pickle.load(f)
else: results = nested_dict()


# In[16]:


# Create Datamodules for all datasets
german_credit_datamodule = GermanCreditDatamodule(data_dir='data/', batch_size=10)
compas_datamodule = CompasDatamodule(data_dir='data/', batch_size=50)
drug_consumption_datamodule = DrugConsumptionDatamodule(data_dir='data/', batch_size=10)

for dm in [german_credit_datamodule, compas_datamodule, drug_consumption_datamodule]:
    for method in ['Solans et al']:
        print(f'Poisoning {dm.get_dataset_name()} dataset with {method} attack:')
        for eps in tqdm(np.arange(0, 1.1, 0.1)):
            pl.seed_everything(123)
            
            metrics = {'test_error': [], 'SPD': [], 'EOD': []}
            for run in range(3):
                # Create a Binary Classifier model for each dataset
                model = BinaryClassifier('LogisticRegression', dm.get_input_size(), lr=1e-3)
                print(f"\tEpsilon: {eps}")

                # Create poisoned dataset
                if eps == 0:
                    dm_poisoned = dm
                else:
                    dm_poisoned = attack(dm, model, eps, method)

                # Crate trainer
                trainer = pl.Trainer(
                    max_epochs=100,
                    gpus=1 if torch.cuda.is_available() else 0,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                    log_every_n_steps=1,
                    callbacks=[EarlyStopping(monitor="train_acc", mode="max", patience=10)]
                )
                
                # Train on the poisoned dataset
                trainer.fit(model, dm_poisoned)
                
                # Save Accuracy and Fairness metrics
                run_results = trainer.test(model, dm)[0]
                for metric, value in run_results.items():
                    metrics[metric].append(value)
                
            # Save mean of metrics
            results[dm.get_dataset_name()]['Test Error'][method]['mean'][eps] = np.mean(metrics['test_error'])
            results[dm.get_dataset_name()]['Statistical Parity'][method]['mean'][eps] = np.mean(metrics['SPD'])
            results[dm.get_dataset_name()]['Equality of Opportunity'][method]['mean'][eps] = np.mean(metrics['EOD'])
            
            # Save standard deviation of metrics
            results[dm.get_dataset_name()]['Test Error'][method]['std'][eps] = np.std(metrics['test_error'])
            results[dm.get_dataset_name()]['Statistical Parity'][method]['std'][eps] = np.std(metrics['SPD'])
            results[dm.get_dataset_name()]['Equality of Opportunity'][method]['std'][eps] = np.std(metrics['EOD'])
        
            with open('/home/lcur0488/different_epsilons.pkl', 'wb') as f:
                pickle.dump(results, f)
            


# and plot results:
