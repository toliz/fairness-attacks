import argparse
import csv
import logging
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import Tensor, IntTensor, BoolTensor
from torch.nn import BCEWithLogitsLoss

import utils
from attacks import influence_attack, anchoring_attack
from datamodules import Dataset, Datamodule, GermanCreditDatamodule, CompasDatamodule, DrugConsumptionDatamodule
from fairness import FairnessLoss
from trainingmodule import BinaryClassifier

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def create_poisoned_dataset(
    args: argparse.Namespace,
    dm: Datamodule,
    model: Optional[BinaryClassifier]
) -> Dataset:
    """
    Create the poisoned dataset based on the provided arguments.
    
    Args:
        args: the arguments from parser
        dm: the datamodule with the clean dataset
        model: the model to get the gradients for, when needed

    Returns: the poisoned dataset
    """
    if args.attack in ['IAF', 'Koh', 'Solans']:
        bce_loss, fairness_loss = BCEWithLogitsLoss(), FairnessLoss(dm.get_sensitive_index())

        if args.attack == 'IAF':
            # Create adversarial loss according to Mehrabi et al.
            def adv_loss(m: BinaryClassifier, X: Tensor, Y: IntTensor, _: BoolTensor):
                return bce_loss(m(X), Y.float()) + 1.0 * fairness_loss(X, *m.get_params())
        elif args.attack == 'Koh':
            # Create adversarial loss according to Koh et al.
            def adv_loss(m: BinaryClassifier, X: Tensor, Y: IntTensor, _: BoolTensor):
                return bce_loss(m(X), Y.float())
        else:
            # Create adversarial loss according to Solans et al.
            def adv_loss(m: BinaryClassifier, X: Tensor, Y: IntTensor, adv_mask: BoolTensor):
                lamda = torch.sum(~adv_mask) / torch.sum(adv_mask)
                return bce_loss(m(X[~adv_mask]), Y[~adv_mask].float()) \
                    + lamda * bce_loss(m(X[adv_mask]), Y[adv_mask].float())

        # Create new training pipeline to use in influence attack
        trainer = pl.Trainer(
            max_epochs=args.epochs,
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
            eps=args.eps,
            eta=args.eta,
            attack_iters=args.attack_iters,
        )
    elif args.attack in ('RAA', 'NRAA'):
        poisoned_dataset = anchoring_attack(
            D_c=dm.get_train_dataset(),
            sensitive_idx=dm.get_sensitive_index(),
            eps=args.eps,
            tau=args.tau,
            sampling_method='random' if args.attack == 'RAA' else 'non-random',
            distance_norm=args.distance_norm,
            distances_type=args.distances_type,
        )
    else:
        raise ValueError(f'Unknown attack {args.attack}.')

    return poisoned_dataset


def main(args: argparse.Namespace):
    """
    Executes the main functionality of the framework, which is to run an attack scenario based on the arguments passed.

    Args:
        args: the arguments passed to the program
    """
    pl.seed_everything(123)
    test_results = []
    for run in range(args.num_runs):
        # Set up the Datamodule based on the dataset name
        if args.dataset == 'German_Credit':
            dm = GermanCreditDatamodule(args.path, args.batch_size)
        elif args.dataset == 'COMPAS':
            dm = CompasDatamodule(args.path, args.batch_size)
        elif args.dataset == 'Drug_Consumption':
            dm = DrugConsumptionDatamodule(args.path, args.batch_size)
        else:
            raise ValueError(f'Unknown dataset {args.dataset}.')

        model = BinaryClassifier(args.model, dm.get_input_size(), lr=1e-3, fairness_metrics_abs=args.fairness_metrics_abs)

        # Create a training pipeline for the model
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            gpus=1 if torch.cuda.is_available() else 0,
            callbacks=[TQDMProgressBar(), EarlyStopping(monitor="train_acc", mode="max", patience=10)]
        )

        # Poison the training set
        if args.attack != 'None' and args.eps > 0:
            poisoned_dataset = create_poisoned_dataset(args, dm, model)
            dm.update_train_dataset(poisoned_dataset)

        # Train
        trainer.fit(model, dm)

        # Test
        test_results.append(*trainer.test(model, dm))

    # Compute average results
    avg_results = utils.get_average_results(test_results, args.num_runs)
    avg_results['name'] = utils.create_experiment_name(args)

    # Write a csv with the results
    with open('results.csv', 'a') as file:
        w = csv.DictWriter(file, avg_results.keys())
        if file.tell() == 0:
            w.writeheader()
        w.writerow(avg_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Datamodule
    parser.add_argument('--dataset',
                        default='German_Credit',
                        type=str,
                        choices=['German_Credit', 'COMPAS', 'Drug_Consumption'],
                        help='Dataset to use')
    parser.add_argument('--path',
                        default='data/',
                        type=str,
                        help='Path to find or save dataset')
    parser.add_argument('--batch_size',
                        default=10,
                        type=int,
                        help='Batch size to use')

    # Training
    parser.add_argument('--model',
                        default='LogisticRegression',
                        type=str,
                        choices=['LogisticRegression'],
                        help='Model to use')
    parser.add_argument('--epochs',
                        default=300,
                        type=int,
                        help='Number of epochs for training')
    parser.add_argument('--num_runs',
                        default=1,
                        type=int,
                        help='Number of runs with different seeds')

    # Attacks
    parser.add_argument('--attack',
                        default='IAF',
                        type=str,
                        choices=['None', 'IAF', 'RAA', 'NRAA', 'Koh', 'Solans'],
                        help='Attack to use')
    parser.add_argument('--eps',
                        default=0.1,
                        type=float,
                        help='Percentage of poisoned data to generate compared to clean data')

    # Anchoring Attack
    parser.add_argument('--tau',
                        default=0,
                        type=float,
                        help='Vicinity distance of perturbed adversarial points')
    parser.add_argument('--distance_norm',
                        default='l1',
                        type=str,
                        choices=['l1', 'l2'],
                        help='Distance norm to use')
    parser.add_argument('--distances_type',
                        default='exp',
                        type=str,
                        choices=['exp', 'percentile'],
                        help='Distance type to use')

    # Influence Attack
    parser.add_argument('--lamda',
                        default=1,
                        type=float,
                        help='Regularization term for fairness loss in influence attack.')
    parser.add_argument('--eta',
                        default=0.01,
                        type=float,
                        help='Step size for gradient update in influence attack')
    parser.add_argument('--attack_iters',
                        default=100,
                        type=int,
                        help='Attack iterations')
    # metrics
    parser.add_argument('--fairness_metrics_abs',
                        default=True,
                        type=bool,
                        help='Use the absolute value of the fairness metrics.')

    program_args = parser.parse_args()

    main(program_args)
