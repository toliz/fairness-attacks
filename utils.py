from attacks.anchoringattack import AnchoringAttackDatamodule
from attacks.datamodule import DataModule
import collections
import functools
import operator


def create_datamodule(args):
    if args.attack == 'None':
        return DataModule(dataset=args.dataset, path=args.path, batch_size=args.batch_size)
    elif args.attack == 'Anchoring':
        return AnchoringAttackDatamodule(dataset=args.dataset, path=args.path, batch_size=args.batch_size,
                                         method=args.anchoring_method, epsilon=args.epsilon, tau=args.tau)
    elif args.attack == 'Influence':
        raise NotImplementedError("Influence attack not implemented.")


def get_average_results(results: list, num_runs: int):
    average_results = dict(functools.reduce(operator.add, map(collections.Counter, results)))
    average_results = {k: v / num_runs for k, v in average_results.items()}
    return average_results

