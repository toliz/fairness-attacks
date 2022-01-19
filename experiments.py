from models.mlp import MLP
from trainingmodule import Classifier
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from utils import create_datamodule
from utils import get_average_results
import csv
import numpy as np
from argparse import Namespace
from attacks.influenceattack import InfluenceAttackDatamodule


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
        model = Classifier(model=MLP(input_size=dm.get_input_size(),
                                     num_hidden=16,
                                     num_classes=dm.get_num_classes()),
                           dm=dm)

        # Call the model with the lowest val_loss at the end of the training
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='model/model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min')

        # Set the logger
        wandb_logger = WandbLogger()
        wandb.init(entity="angelosnal",
                   project=args.project,
                   job_type='train',
                   name=args.experiment,
                   mode=args.logger_mode,
                   notes=args.logger_notes)

        # Set the trainer
        trainer = pl.Trainer(
            # max_epochs=args.epochs,
            max_epochs=1,
            progress_bar_refresh_rate=1,
            gpus=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )

        # Train and Test
        # TODO: for t > 1
        if isinstance(dm, InfluenceAttackDatamodule):
            for i in range(10):
                trainer.fit(model, dm)
                dm.update_dataset(model)
        else:
            trainer.fit(model, dm)
        test_results.append(*trainer.test(model, dm))
        wandb.finish()

    # Compute average results
    avg_results = get_average_results(test_results, args.num_runs)
    avg_results['name'] = args.experiment

    # write csv to memory
    # TODO: reformat saving headers
    with open('results.csv', 'a') as file:
        w = csv.DictWriter(file, avg_results.keys())
        if file.tell() == 0:
            w.writeheader()
        w.writerow(avg_results)

    # log results to wandb
    if args.log_average_resutls:
        wandb.init(entity="angelosnal",
                   project=args.project,
                   job_type='train',
                   name='Average: ' + args.experiment,
                   notes=args.logger_notes)
        wandb.log(avg_results)
        wandb.finish()


def run_all_experiments():
    args = {
        'path': 'data/',
        'batch_size': 64,
        'model': 'mlp',
        'epochs': 20,
        'num_runs': 5,
        'tau': 0,
        'project': 'FACT_AI',
        'logger_notes': '',
        'logger_mode': 'disabled',
        'entity': 'angelosnal',
        'log_average_resutls': True,
    }
    datasets = ['German_Credit']
    attacks = ['Anchoring']
    anchoring_methods = ['random', 'non_random']
    epsilons = list(np.linspace(0.1, 1, 10))

    for dataset in datasets:
        for attack in attacks:
            for anchoring_method in anchoring_methods:
                for epsilon in epsilons:
                    experiment_name = '-'.join([dataset, attack, anchoring_method, str(epsilon)])
                    args['dataset'] = dataset
                    args['attack'] = attack
                    args['anchoring_method'] = anchoring_method
                    args['epsilon'] = epsilon
                    args['experiment'] = experiment_name
                    run_experiment(Namespace(**args))


if __name__ == '__main__':
    run_all_experiments()
