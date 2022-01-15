from attacks.anchoringattack import AnchoringAttackDatamodule
from attacks.datamodule import DataModule
import collections
import functools
import operator
import pytorch_lightning as pl


def create_datamodule(args) -> pl.LightningDataModule:
    '''
    Return pytorch lightning datamodule
    '''
    if args.attack == 'None':
        return DataModule(dataset=args.dataset, path=args.path, batch_size=args.batch_size)
    elif args.attack == 'Anchoring':
        return AnchoringAttackDatamodule(dataset=args.dataset, path=args.path, batch_size=args.batch_size,
                                         method=args.anchoring_method, epsilon=args.epsilon, tau=args.tau,
                                         )
    elif args.attack == 'Influence':
        raise NotImplementedError("Influence attack not implemented.")


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

