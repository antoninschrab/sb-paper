# Experiments for Aggregation of Statistical Evidence under Exchangeability

This repository provides the implementation for the following hypothesis tests in [tests.py](tests.py): SB (Single-Step), TB (Two-Batch), SeqSB (Sequential Single-Step) as well as their 'worst-case' variants for minimum (Bonferroni), average and median merging functions. It also provides implementation for the TB and SB conformal prediction procedures, presented in [conformal.py](conformal.py) These tests can be used with any statistics (provided by the user), and efficient implementations for HSIC and MMD are provided. The code is available under the [MIT License](LICENSE.md).

The required dependencies can be installed using the provided Conda environment files. For standard CPU execution, run:
```bash
conda env create -f env_cpu.yml
conda activate sb-env
```
For standard GPU execution (benefitting from JAX hardware acceleration), run:
```bash
conda env create -f env_gpu.yml
conda activate sb-env
```

The code to reproduce the experiments of the paper 'Aggregation of Statistical Evidence under Exchangeability' can be found in [experiments.ipynb](experiments.ipynb). Running this notebook saves all the results in the [results](results) directory as `.npz` files. The figures, of small or large size, can then be generated with the code in [figures_small.ipynb](figures_small.ipynb) or [figures_large.ipynb](figures_large.ipynb) which saves them as `.pdf` files in the directory [figures_small](figures_small) or [figures_large](figures_large).
