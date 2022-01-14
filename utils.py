from attacks.anchoringattack import AnchoringAttackDatamodule
from datamodule import DataModule


def create_datamodule(args):
    if args.attack == 'None':
        return DataModule(dataset=args.dataset, path=args.path, batch_size=args.batch_size)
    elif args.attack == 'Anchoring':
        return AnchoringAttackDatamodule(dataset=args.dataset, path=args.path, batch_size=args.batch_size,
                                         method='random', epsilon=1, tau=1)
    elif args.attack == 'Influence':
        raise NotImplementedError("Influence attack not implemented.")
