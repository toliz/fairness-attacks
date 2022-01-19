import argparse
from experiments import run_experiment, run_all_experiments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DataModule
    parser.add_argument('--dataset',
                        default='German_Credit',
                        type=str,
                        choices=['German_Credit', 'Drug_Consumption', 'COMPAS'],
                        help='Dataset name to use')
    parser.add_argument('--path',
                        default='data/',
                        type=str,
                        help='Path to find or save dataset')
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='Batch size to use')

    # Training
    parser.add_argument('--model',
                        default='MLP',
                        type=str,
                        choices=['MLP'],
                        help='Model name to use')
    parser.add_argument('--epochs',
                        default=20,
                        type=int,
                        help='Number of epochs for training')
    parser.add_argument('--num_runs',
                        default=5,
                        type=int,
                        help='Number of runs with different initializations')

    # Attacks
    parser.add_argument('--attack',
                        default='Influence',
                        type=str,
                        choices=['Anchoring', 'Influence', 'None'],
                        help='Name of the attack')

    # Anchoring Attack
    parser.add_argument('--anchoring_method',
                        default='non_random',
                        type=str,
                        choices=['random', 'non_random'],
                        help='Sampling method for anchoring attack')
    parser.add_argument('--tau', default='0', type=float, help='')
    parser.add_argument('--epsilon', default='1', type=float, help='')

    # Influence Attack
    parser.add_argument('--fairness_loss',
                        default='sensitive_cov_boundary',
                        type=str,
                        choices=['sensitive_cov_boundary'],
                        help='Fairness loss to be used in influence attack')
    parser.add_argument('--fairness_reg_term',
                        default='1',
                        type=float,
                        help='Regularization term for fairness loss in influence attack.')
    parser.add_argument('--influence_step_size',
                        default='0.01',
                        type=float,
                        help='Step size for gradient update in influence attack')

    # Logger
    parser.add_argument('--project',
                        default='FACT_AI',
                        type=str,
                        help='Project name to save the logs')
    parser.add_argument('--experiment',
                        default='mlp_baseline',
                        type=str,
                        help='Experiment name to save the logs')
    parser.add_argument('--logger_notes',
                        default='',
                        type=str,
                        help='Description of the experiment')
    parser.add_argument('--logger_mode',
                        default='disabled',
                        type=str,
                        choices=['online', 'offline', 'disabled'],
                        help='Mode of logger')
    parser.add_argument('--entity',
                        default='angelosnal',
                        type=str,
                        help='Owner\' username of wandb project')
    parser.add_argument('--log_average_resutls',
                        default=True,
                        type=bool,
                        help='Log the average results from the runs')

    # Experiment
    parser.add_argument('--run_all_experiments',
                        default=False,
                        type=bool,
                        help='If true, all experiments will run')


    args = parser.parse_args()

    if args.run_all_experiments:
        run_all_experiments()
    else:
        run_experiment(args)
