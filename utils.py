from attacks.anchoringattack import AnchoringAttackDatamodule
from attacks.datamodule import DataModule


def create_datamodule(args):
    if args.attack == 'None':
        return DataModule(dataset=args.dataset, path=args.path, batch_size=args.batch_size)
    elif args.attack == 'Anchoring':
        return AnchoringAttackDatamodule(dataset=args.dataset, path=args.path, batch_size=args.batch_size,
                                         method=args.anchoring_method, epsilon=args.epsilon, tau=args.tau)
    elif args.attack == 'Influence':
        raise NotImplementedError("Influence attack not implemented.")
