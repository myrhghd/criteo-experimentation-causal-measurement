# Criteo Experimentation and Causal Measurement

## Business problem

Advertising teams need to distinguish incremental customer behavior caused by a campaign from outcomes that would have occurred without treatment. Reliable measurement supports budget allocation and audience decisions.

## Analytical objective

This project measures average treatment effects from a randomized advertising experiment. Future work will estimate heterogeneous treatment effects and evaluate uplift models using conditional average treatment effects.

## Dataset

The project uses the public `CRITEO-UPLIFTv2.1` dataset from the [official Criteo dataset page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) and its [large scale individual treatment effect and uplift modeling benchmark](https://github.com/criteo-research/large-scale-ITE-UM-benchmark). The current Criteo page states that the dataset uses the [Creative Commons Attribution NonCommercial ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/) and requests citation of Diemert et al. (2018), *A Large Scale Benchmark for Uplift Modeling*.

The `treatment` column records randomized treatment assignment, while `exposure` records effective advertising exposure after assignment. Treatment assignment is the primary treatment for subsequent causal analysis.

## Analytical scope

The current analysis covers data validation, randomization diagnostics, intention to treat effect estimation, and experiment sensitivity. Heterogeneous treatment effect estimation, uplift modeling, and model evaluation remain future work.

## Experiment results

The released benchmark contains 13,979,592 observations. Treatment assignment was 85.000013%, compared with the documented 85% design. The maximum absolute covariate standardized difference was 0.036, below the 0.10 practical threshold. A treatment prediction diagnostic produced a ROC AUC of 0.5046 and log loss of 0.4237, compared with baseline log loss of 0.4217.

| Outcome | Control rate | Treatment rate | ITT effect | 95% CI | Relative lift | MDE 80% | MDE 90% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Visit | 3.820% | 4.854% | +1.034 pp | [1.006, 1.063] pp | 27.1% | 0.040 pp | 0.047 pp |
| Conversion | 0.194% | 0.309% | +0.115 pp | [0.108, 0.122] pp | 59.4% | 0.009 pp | 0.011 pp |

The randomization diagnostics are consistent with the documented design. Treatment increased visits and conversions in the released benchmark population, and both effects exceed the corresponding minimum detectable effects. These estimates do not represent original Criteo campaign return on investment or production incrementality.

Run the complete experiment analysis and regenerate its figures:

```bash
criteo-analysis experiment
```

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
