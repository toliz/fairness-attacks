import argparse
import csv
import pytorch_lightning as pl
import torch
import utils
import wandb

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.nn import BCEWithLogitsLoss

from attacks import influence_attack, anchoring_attack
from datamodules import GermanCreditDatamodule, CompasDatamodule, DrugConsumptionDatamodule
from fairness import FairnessLoss
from trainingmodule import BinaryClassifier
from datamodules.datamodule import Datamodule


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
    if args.attack == 'IAF':
        # Create adversarial loss
        bce_loss, fairness_loss = BCEWithLogitsLoss(), FairnessLoss(dm.get_sensitive_index())
        adv_loss = lambda _model, X, y: (
                bce_loss(_model(X), y.float()) + args.lamda * fairness_loss(X, *_model.get_params())
        )
        
        # Create new training pipeline to use in influence attack
        trainer = pl.Trainer(
            max_epochs=100,
            gpus=1 if torch.cuda.is_available() else 0,
            callbacks=[EarlyStopping(monitor="train_loss", mode="min")]
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
            attack_iters=args.attack_iters,
        )
    else:
        raise ValueError(f'Unknown attack {args.attack}.')

    return poisoned_dataset


def main(args: argparse.Namespace):
    pl.seed_everything(123)
    test_results = []
    for run in range(args.num_runs):
        # Set-up W&B logger
        wandb_logger = WandbLogger()
        wandb.init(
            entity=args.wandb_user,
            project=args.logger_project,
            job_type='train',
            name=utils.create_experiment_name(args) + f'run_{run}',
            mode=args.logger_mode,
            notes=args.logger_notes
        )

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
            logger=wandb_logger,
            callbacks=[TQDMProgressBar()]
        )
        
        # Poison the training set
        if args.attack != 'None':
            poisoned_dataset = create_poisoned_dataset(args, dm, model)
            dm.update_train_dataset(poisoned_dataset)
            
        # Train
        trainer.fit(model, dm)
        
        # Test
        test_results.append(*trainer.test(model, dm))
        wandb.finish()
    
    # Compute average results
    avg_results = utils.get_average_results(test_results, args.num_runs)
    avg_results['name'] = utils.create_experiment_name(args)

    # write csv to memory
    # TODO: reformat saving headers
    with open('results.csv', 'a') as file:
        w = csv.DictWriter(file, avg_results.keys())
        if file.tell() == 0:
            w.writeheader()
        w.writerow(avg_results)

    # log results to wandb
    if args.log_average_results:
        wandb.init(entity=args.wandb_user,
                   project=args.project,
                   job_type='train',
                   name='Average: ' + utils.create_experiment_name(args),
                   notes=args.logger_notes)
        wandb.log(avg_results)
        wandb.finish()


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
                        default=100,
                        type=int,
                        help='Batch size to use')

    # Training
    parser.add_argument('--model',
                        default='LogisticRegression',
                        type=str,
                        choices=['LogisticRegression'],
                        help='Model to use')
    parser.add_argument('--epochs',
                        default=1,
                        type=int,
                        help='Number of epochs for training')
    parser.add_argument('--num_runs',
                        default=1,
                        type=int,
                        help='Number of runs with different seeds')

    # Attacks
    parser.add_argument('--attack',
                        default='RAA',
                        type=str,
                        choices=['None', 'IAF', 'RAA', 'NRAA'],
                        help='Attack to use')
    parser.add_argument('--eps',
                        default=0.1,
                        type=float,
                        help='Percentage of poisoned data to generate compared to clean data')
    parser.add_argument('--attack_iters',
                        default=10000,
                        type=int,
                        help='Attack iterations')

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

    # Logger
    parser.add_argument('--logger_project',
                        default='FACT_AI',
                        type=str,
                        help='Project name to save the logs')
    parser.add_argument('--logger_notes',
                        default='',
                        type=str,
                        help='Description of the experiment')
    parser.add_argument('--logger_mode',
                        default='disabled',
                        type=str,
                        choices=['online', 'offline', 'disabled'],
                        help='Mode of logger')
    parser.add_argument('--wandb_user',
                        default='angelosnal',
                        type=str,
                        help='Wandb project owner username')
    parser.add_argument('--log_average_results',
                        default=False,
                        type=bool,
                        help='Log the average results from the runs')

    args = parser.parse_args()

    main(args)
