# [Re] Exacerbating Algorithmic Bias through Fairness Attacks

This repository is a re-implementation of [Exacerbating Algorithmic Bias through Fairness Attacks](https://arxiv.org/abs/2012.08723).

## Requirements

To set up the development environment with conda:

```setup
conda env create -f gpu_env.yml
```

The datasets used for the experiments are:

- [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- [COMPAS Dataset](https://github.com/propublica/compas-analysis)
- [Drug Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)

There is no need to manually download them, as they will be fetched automatically when loading the datamodules.

## Attacks

Currently, the following attacks are implemented:

- Random Anchoring Attack (RAA)
- Non-Random Anchoring Attack (NRAA)
- Influence Attack on Fairness (IAF)
- Influence Attack [(Koh et al.)](https://arxiv.org/abs/1703.04730)
- Poisoning Attack Against Algorithmic Fairness [(Solans et al.)](https://arxiv.org/abs/2004.07401)

The launch arguments for the experiments script are:

### Dataset - Datamodules

- `--dataset`: one of *German_Credit*, *COMPAS*, *Drug_Consumption* (default: `German_Credit`)
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
- `--distance_norm`: the distance norm to use for neighbor calculations; one of *l1*, *l2* (default: `l1`)
- `--distances_type`: the type of distances used to identify the most popular point in the NRAA; one of *exp*, *percentile* (default: `exp`)

#### Influence Attack

- `--fairness-loss`: the fairness loss function to use; one of `sensitive_cov_boundary`, `TODO: add more` (default: `sensitive_cov_boundary`)
- `--lamda`: the regularization term to use between performance and fairness impact (default: `1`)
- `--eta`: step size for the gradient update step (default: `0.01`)
- `--attack_iters`: the amount of EM iterations to perform (default: `100`)

To perform an attack, run this command with the desired arguments:

```train
python main.py --dataset COMPAS --batch_size 20 ...
```

## Experiments

The experiments, which reproduce both the work in the original paper and our novel ideas, are located in the [experiments](experiments/) directory as a set of Jupyter Notebooks. More specifically:

- [different-attacks](experiments/different-attacks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/toliz/FACT-AI/blob/main/experiments/different-attacks.ipynb)  reproduces *Figure 2* from the original paper,
- [different-lambda](experiments/different-lambda.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/toliz/FACT-AI/blob/main/experiments/different-lambda.ipynb) reproduces *Figure 3* from the original paper, and
- [augmentation](experiments/augmentation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/toliz/FACT-AI/experiments/different-lambda.ipynb) tests out whether using a negative value for the regularization term can lead to a less biased dataset. # Fix links

## Results

A summary of the results, without the need for model training (which took several days on a Server with a GTX 1080 GPU), can be generated with the notebook [...](?).

[Results](results/plot_results.ipynb).

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.
