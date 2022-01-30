import argparse
import csv
import logging
import pytorch_lightning as pl
import torch
import utils

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn import BCEWithLogitsLoss

from attacks import influence_attack, anchoring_attack
from datamodules import Datamodule, GermanCreditDatamodule, CompasDatamodule, DrugConsumptionDatamodule
from fairness import FairnessLoss
from trainingmodule import BinaryClassifier


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def create_poisoned_dataset(
    args: argparse.Namespace,
    dm: Datamodule,
    model: BinaryClassifier,
):
    """
    Function that returns the poisoned dataset
    
    Args:
        args: arguments from parser
        dm: datamodule with the clean dataset
        model: model to get the gradients for influence attack

    Returns: the poisoned dataset
    """
    if args.attack in ['IAF', 'Koh', 'Solans']:
        bce_loss, fairness_loss = BCEWithLogitsLoss(), FairnessLoss(dm.get_sensitive_index())
        
        if args.attack == 'IAF':
            # Create adversarial loss according to Mehrabi et al.
            adv_loss = lambda _model, X, y, adv_mask: (
                    bce_loss(_model(X), y.float()) + 1.0 * fairness_loss(X, *_model.get_params())
            )
        elif args.attack == 'Koh':
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
    pl.seed_everything(123)
    test_results = []
    for run in range(args.num_runs):

        # Set-up PyTorch Lightning    
        if args.dataset == 'German_Credit':
            dm = GermanCreditDatamodule(args.path, args.batch_size)
        elif args.dataset == 'COMPAS':
            dm = CompasDatamodule(args.path, args.batch_size)
        elif args.dataset == 'Drug_Consumption':
            dm = DrugConsumptionDatamodule(args.path, args.batch_size)
        else:
            raise ValueError(f'Unknown dataset {args.dataset}.')
        
        model = BinaryClassifier(args.model, dm.get_input_size(), lr=1e-3)

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

    # Write csv with the results to memory
    with open('results.csv', 'a') as file:
        w = csv.DictWriter(file, avg_results.keys())
        if file.tell() == 0:
            w.writeheader()
        w.writerow(avg_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # DataModule
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
    parser.add_argument('--attack_iters',
                        default=100,
                        type=int,
                        help='Attack iterations')
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

    # Anchoring Attack
    parser.add_argument('--tau',
                        default=0,
                        type=float,
                        help='Vicinity distance of perturbed adversarial points')

    # Influence Attack
    parser.add_argument('--fairness_loss',
                        default='sensitive_cov_boundary',
                        type=str,
                        choices=['sensitive_cov_boundary'],
                        help='Fairness loss to be used in influence attack')
    parser.add_argument('--lamda',
                        default=1,
                        type=float,
                        help='Regularization term for fairness loss in influence attack.')
    parser.add_argument('--eta',
                        default=0.01,
                        type=float,
                        help='Step size for gradient update in influence attack')

    args = parser.parse_args()

    main(args)
