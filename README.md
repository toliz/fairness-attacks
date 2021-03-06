# [Re] Exacerbating Algorithmic Bias through Fairness Attacks

This repository is a re-implementation of the paper [Exacerbating Algorithmic Bias through Fairness Attacks](https://arxiv.org/abs/2012.08723). For reference, the original codebase can be found [here](https://github.com/Ninarehm/attack).

## Requirements

To set up the development environment with conda:

```setup
conda env create -f environment.yml
```

## Datasets

The datasets used for the experiments are:

- [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- [COMPAS Dataset](https://github.com/propublica/compas-analysis)
- [Drug Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)

There is no need to manually download them, as they will be fetched automatically when loading the datamodules.

## Attacks

Currently, the following attacks are implemented:

- Random Anchoring Attack (RAA) [(Ninareh et al.)](https://arxiv.org/abs/2012.08723)
- Non-Random Anchoring Attack (NRAA) [(Ninareh et al.)](https://arxiv.org/abs/2012.08723)
- Influence Attack on Fairness (IAF) [(Ninareh et al.)](https://arxiv.org/abs/2012.08723)
- Influence Attack [(Koh et al.)](https://arxiv.org/abs/1703.04730)
- Poisoning Attack Against Algorithmic Fairness [(Solans et al.)](https://arxiv.org/abs/2004.07401)

The launch arguments for the experiments script are:

### Dataset

- `--dataset`: one of `German_Credit`, `COMPAS`, `Drug_Consumption` (default: `German_Credit`)
- `--path`: the path where the dataset will be saved (default: `./data/`)
- `--batch_size`: the batch size to use for the train and test dataloaders (default: `10`)

### Training Pipeline

- `--model`: the model that the defender will be using; currently, only `LogisticRegression` is supported (default: `LogisticRegression`)
- `--epochs`: the max epochs to use for training - early stopping based on train accuracy is also applied (default: `300`)
- `--num_runs`: the number of runs will be executed with different seeds (default: `1`)

### Attack Parameters

- `--attack`: the attack to perform; one of `None`, `RAA`, `NRAA`, `IAF`, `Koh`, `Solans` (default: `NRAA`)
- `--eps`: the percentage of poisoned data to generate, compared to clean data (default: `0.1`)

#### Anchoring Attack

- `--tau`: the maximum acceptable distance from x_target for the perturbed points (default: `0`)
- `--distance_norm`: the distance norm to use for neighbor calculations; one of `l1`, `l2` (default: `l1`)
- `--distances_type`: the type of distances used to identify the most popular point in the NRAA; one of *exp*, *percentile* (default: `exp`)

#### Influence Attack

- `--lamda`: the regularization term to use between performance and fairness impact (default: `1`)
- `--eta`: step size for the gradient update step (default: `0.01`)
- `--attack_iters`: the amount of EM iterations to perform (default: `100`)

To perform an attack, run this command with the desired arguments:

```train
python main.py --dataset COMPAS --batch_size 20 ...
```

## Experiments

The experiments, which reproduce both the work in the original paper and our novel ideas, are located in the [experiments](experiments/) directory as a set of Jupyter Notebooks. More specifically:

- [different-attacks](experiments/different-attacks.ipynb)  reproduces our first experiment, as shown below
- [different-lambda](experiments/different-lambda.ipynb) reproduces our second experiment, as shown below
- [augmentation](experiments/augmentation.ipynb) reproduces our third experiment, as shown below

## Results

A summary of the results, without the need for model training (which took several days on a Server with a GTX 1080 GPU), can be generated with the notebook [results](results/plot_results.ipynb).

![Experiment 1](results/exp-1.png)

![Experiment 2](results/exp-2.png)

![Experiment 3](results/exp-3.png)

## Contributing

This project is licensed under the MIT License.