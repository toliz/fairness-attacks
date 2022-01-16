import pandas as pd
from attacks.anchoringattack import AnchoringAttackDatamodule
from attacks.datamodule import DataModule
import collections
import functools
import operator
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np


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


# TODO: reformat
def plot_results():
    df = pd.read_csv('results.csv')
    random_eod = df.iloc[0:10].EOD.values
    random_spd = df.iloc[0:10].SPD.values
    random_acc = df.iloc[0:10].test_acc.values

    nrandom_eod = df.iloc[10:20].EOD.values
    nrandom_spd = df.iloc[10:20].SPD.values
    nrandom_acc = df.iloc[10:20].test_acc.values

    epsilons = np.linspace(0.1, 1, 10)
    print(epsilons)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

    ax[0].plot(epsilons, 1 - random_acc, linestyle='--', marker='o', label='RAA')
    ax[0].plot(epsilons, 1 - nrandom_acc, linestyle='--', marker='x', label='NRAA')
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].plot(epsilons, random_spd, linestyle='--', marker='o', label='RAA')
    ax[1].plot(epsilons, nrandom_spd, linestyle='--', marker='x', label='NRAA')
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel('SPD')
    ax[1].legend()

    ax[2].plot(epsilons, random_eod, linestyle='--', marker='o', label='RAA')
    ax[2].plot(epsilons, nrandom_eod, linestyle='--', marker='x', label='NRAA')
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel('EOD')
    ax[2].legend()


    fig.suptitle('German Credit')
    plt.tight_layout()
    plt.show()



plot_results()