# Criteo Experimentation and Causal Measurement

## Business problem

Advertising teams need to distinguish incremental customer behavior caused by a campaign from outcomes that would have occurred without treatment. Reliable measurement supports budget allocation and audience decisions.

## Analytical objective

This project will measure average and individual treatment effects from a randomized advertising experiment. It will also evaluate whether uplift models can identify customers whose outcomes are most likely to change because of treatment.

## Dataset

The project uses the public `CRITEO-UPLIFTv2.1` dataset from the [Criteo Research large scale individual treatment effect and uplift modeling benchmark](https://github.com/criteo-research/large-scale-ITE-UM-benchmark). The expected download filename is `criteo-research-uplift-v2.1.csv.gz`.

## Analytical scope

The planned analysis covers data validation, randomization and experiment diagnostics, A/B test analysis, statistical power, causal treatment effect estimation, uplift modeling, and model evaluation.

## Local setup

```bash
conda env create --file environment.yml
conda activate criteo-experiment
python -m pip install --no-deps --editable .
pytest
```
