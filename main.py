import argparse
from models.mlp import MLP
from trainingmodule import Classifier
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from utils import create_datamodule
from utils import get_average_results


def run_experiment(args):
    seed_everything(123)

    # Save the results from every run
    test_results = []
    for i in range(args.num_runs):

        # Set the data module
        dm = create_datamodule(args)
        dm.prepare_data()
        dm.setup()

        # Set the model
        model = Classifier(model=MLP(input_size=dm.get_input_size(), num_hidden=16, num_classes=dm.get_num_classes()),
                           dm=dm)

        # Call the model with the lowest val_loss at the end of the training
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='model/model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min')

        # Set the logger
        wandb_logger = WandbLogger()
        wandb.init(entity="angelosnal", project=args.project, job_type='train', name=args.experiment,
                   mode=args.logger_mode, notes=args.logger_notes)

        # Set the trainer
        trainer = pl.Trainer(max_epochs=args.epochs,
                             progress_bar_refresh_rate=1,
                             gpus=1,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback],
                             )

        # Train and Test
        trainer.fit(model, dm)
        test_results.append(*trainer.test(model, dm))
        wandb.finish()

    avg_results = get_average_results(test_results, args.num_runs)
    if args.log_average_resutls:
        wandb.init(entity="angelosnal", project=args.project, job_type='train', name='Average: '+args.experiment,
                   notes=args.logger_notes)
        wandb.log(avg_results)
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DataModule
    parser.add_argument('--dataset', default='German_Credit', type=str, choices=['German_Credit', 'Drug_Consumption'],
                        help='Dataset name to use')
    parser.add_argument('--path', default='data/', type=str,
                        help='Path to find or save dataset')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size to use')

    # Training
    parser.add_argument('--model', default='MLP', type=str, choices=['MLP'],
                        help='Model name to use')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--num_runs', default=5, type=int,
                        help='Number of runs with different initializations')

    # Attacks
    parser.add_argument('--attack', default='None', type=str, choices=['Anchoring', 'Influence', 'None'],
                        help='Name of the attack')

    # Anchoring Attack
    parser.add_argument('--anchoring_method', default='non_random', type=str, choices=['random', 'non_random'],
                        help='Sampling method for anchoring attack')
    parser.add_argument('--tau', default='0', type=float,
                        help='')
    parser.add_argument('--epsilon', default='1', type=float,
                        help='')

    # Logger
    parser.add_argument('--project', default='FACT_AI', type=str,
                        help='Project name to save the logs')
    parser.add_argument('--experiment', default='mlp_baseline', type=str,
                        help='Experiment name to save the logs')
    parser.add_argument('--logger_notes', default='', type=str,
                        help='Description of the experiment')
    parser.add_argument('--logger_mode', default='disabled', type=str, choices=['online', 'offline', 'disabled'],
                        help='Mode of logger')
    parser.add_argument('--entity', default='angelosnal', type=str,
                        help='Owner\' username of wandb project')
    parser.add_argument('--log_average_resutls', default=True, type=bool,
                        help='Log the average results from the runs')

    args = parser.parse_args()

    run_experiment(args)
