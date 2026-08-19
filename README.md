# Criteo Experimentation and Causal Measurement

## Business problem

Advertising teams need to distinguish incremental customer behavior caused by a campaign from outcomes that would have occurred without treatment. Reliable measurement supports budget allocation and audience decisions.

## Analytical objective

This project will measure average and individual treatment effects from a randomized advertising experiment. It will also evaluate whether uplift models can identify customers whose outcomes are most likely to change because of treatment.

## Dataset

The project uses the public `CRITEO-UPLIFTv2.1` dataset from the [official Criteo dataset page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) and its [large scale individual treatment effect and uplift modeling benchmark](https://github.com/criteo-research/large-scale-ITE-UM-benchmark). The current Criteo page states that the dataset uses the [Creative Commons Attribution NonCommercial ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/) and requests citation of Diemert et al. (2018), *A Large Scale Benchmark for Uplift Modeling*.

The `treatment` column records randomized treatment assignment, while `exposure` records effective advertising exposure after assignment. Treatment assignment is the primary treatment for subsequent causal analysis.

## Analytical scope

The planned analysis covers data validation, randomization and experiment diagnostics, A/B test analysis, statistical power, causal treatment effect estimation, uplift modeling, and model evaluation.

## Local setup

```bash
conda env create --file environment.yml
conda activate criteo-experiment
python -m pip install --no-deps --editable .
pytest
```

## Data preparation

Run the acquisition and preparation commands from the repository root:

```bash
criteo-data acquire
criteo-data prepare
```

The acquisition command streams the official Criteo file to `data/raw/criteo-research-uplift-v2.1.csv.gz`. The preparation command validates the raw data and writes a compressed Parquet file under `data/processed/`. Both data directories remain local and are excluded from Git.
