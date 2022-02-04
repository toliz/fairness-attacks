import argparse
import collections
import functools
import operator
from typing import List


def create_experiment_name(args: argparse.Namespace) -> str:
    """
    Create an experiment name according to the specified arguments `args`.

    Args:
        args: the arguments that define the experiment being run
    """
    experiment_name = f'{args.dataset}_{args.attack}_{args.eps}'
    
    if args.attack == 'IAF':
        experiment_name += f'{args.lamda}'
        
    return experiment_name


def get_average_results(results: List[dict], num_runs: int) -> dict:
    """
    Calculate the average values for each metric in the results.

    Args:
        results: a list of dicts with the results of every run
        num_runs: the number of successful runs that were executed

    Returns: a dict with the average results
    """
    average_results = dict(functools.reduce(operator.add, map(collections.Counter, results)))
    average_results = {k: v / num_runs for k, v in average_results.items()}
    return average_results
