import collections
import functools
import operator


def create_experiment_name(args) -> str:
    experiment_name = f'{args.dataset}_{args.attack}_{args.epsilon}'
    
    if args.attack == 'IAF':
        experiment_name += f'{args.lamda}'
        
    return experiment_name


def get_average_results(results: list, num_runs: int) -> dict:
    '''
    Calculate average metrics of runs
    :param results: list of dicts with the results of every run
    :param num_runs: number of runs
    :return:
    '''
    average_results = dict(functools.reduce(operator.add, map(collections.Counter, results)))
    average_results = {k: v / num_runs for k, v in average_results.items()}
    return average_results
