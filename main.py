import argparse


def main(args: argparse.Namespace):
    raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DataModule
    parser.add_argument('--dataset',
                        default='German_Credit',
                        type=str,
                        choices=['German_Credit', 'Drug_Consumption', 'COMPAS'],
                        help='Dataset to use')
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
                        default='LogisticRegression',
                        type=str,
                        choices=['LogisticRegression'],
                        help='Model to use')
    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='Number of epochs for training')
    parser.add_argument('--num_runs',
                        default=1,
                        type=int,
                        help='Number of runs with different seeds')

    # Attacks
    parser.add_argument('--attack',
                        default='Anchoring',
                        type=str,
                        choices=['Anchoring', 'Influence', 'None'],
                        help='Attack to use')
    parser.add_argument('--epsilon',
                        default=0.1,
                        type=float,
                        help='Percentage of poisoned data to generate compared to clean data')
    parser.add_argument('--attack_iters',
                        default=10000,
                        type=int,
                        help='Attack iterations')

    # Anchoring Attack
    parser.add_argument('--anchoring_method',
                        default='non_random',
                        type=str,
                        choices=['random', 'non_random'],
                        help='Sampling method for anchoring attack')
    parser.add_argument('--tau',
                        default=0,
                        type=float,
                        help='Vicinity distance of pertubed adversarial points')

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
    parser.add_argument('--project',
                        default='FACT_AI',
                        type=str,
                        help='Project name to save the logs')
    parser.add_argument('--logger_notes',
                        default='',
                        type=str,
                        help='Description of the experiment')
    parser.add_argument('--logger_mode',
                        default='online',
                        type=str,
                        choices=['online', 'offline', 'disabled'],
                        help='Mode of logger')
    parser.add_argument('--entity',
                        default='angelosnal',
                        type=str,
                        help='Wandb project owner username')
    parser.add_argument('--log_average_results',
                        default=False,
                        type=bool,
                        help='Log the average results from the runs')

    args = parser.parse_args()

    main(args)
