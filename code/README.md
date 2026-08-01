# Code companion for *Discrete Choice Models*

This directory contains Python implementations, simulations, data-loading scripts, and empirical examples accompanying Alfred Galichon's book *Discrete Choice Models: Mathematical Methods, Econometrics, and Data Science*.

The files are numbered by chapter and example. They are intended for teaching, experimentation, and computational illustration. Some examples require optional software, external data, or substantial computation; see the notes below.

## Installation

Clone the repository and move into its root directory:

```bash
git clone https://github.com/alfredgalichon/DCM.git
cd DCM
```

Create and activate a virtual environment.

### macOS and Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running an example

From the root of the repository, run:

```bash
python code/1-01_market-share-simulation.py
```

Replace the filename with the example you wish to execute.

## Important notes

- Some optimization examples require **Gurobi** and a valid Gurobi licence.
- `4-01_laguerre-diagrams.py` requires the `sdot` package.
- Some scripts retrieve public datasets from the internet and therefore require an active connection.
- Some simulations and estimation routines may take substantial time or memory.
- The code is research and teaching material rather than a packaged software library.
- Exact results may depend on the Python and package versions installed.

## Suggested starting points

These examples provide a useful introduction to several major themes of the book:

| File | Topic |
|---|---|
| [`1-01_market-share-simulation.py`](1-01_market-share-simulation.py) | Simulation of market shares |
| [`2-02_inverting-the-nested-logit-model.py`](2-02_inverting-the-nested-logit-model.py) | Inversion of the nested-logit model |
| [`3-01_logistic-regression-via-gradient-descent.py`](3-01_logistic-regression-via-gradient-descent.py) | Logistic regression by gradient descent |
| [`5-02_ot-via-ipfp.py`](5-02_ot-via-ipfp.py) | Optimal transport using IPFP |
| [`6-03_ddc-logit-finite-estimation-numpy.py`](6-03_ddc-logit-finite-estimation-numpy.py) | Finite-horizon dynamic discrete-choice estimation |

## Contents

### Chapter 1 — Market shares and demand inversion

- [`1-01_market-share-simulation.py`](1-01_market-share-simulation.py)
- [`1-02_demand-inversion-via-lp.py`](1-02_demand-inversion-via-lp.py) — requires Gurobi

### Chapter 2 — Multivariate extreme-value models

- [`2-01_MEV-as-a-factor-model.py`](2-01_MEV-as-a-factor-model.py) — potentially computationally intensive
- [`2-02_inverting-the-nested-logit-model.py`](2-02_inverting-the-nested-logit-model.py)
- [`2-03_convergence-to-max-stable-distribution.py`](2-03_convergence-to-max-stable-distribution.py)

### Chapter 3 — Estimation, regularization, and identification

- [`3-01_logistic-regression-via-gradient-descent.py`](3-01_logistic-regression-via-gradient-descent.py)
- [`3-02_logistic-regression-as-glm.py`](3-02_logistic-regression-as-glm.py)
- [`3-03_coercivity-detector.py`](3-03_coercivity-detector.py) — requires Gurobi
- [`3-04_proximal-gradient-descent.py`](3-04_proximal-gradient-descent.py)
- [`3-05_minimax-regret-estimation.py`](3-05_minimax-regret-estimation.py) — requires Gurobi
- [`3-06_small-noise-limit.py`](3-06_small-noise-limit.py)
- [`3-07_code-for-exercise-3-2.py`](3-07_code-for-exercise-3-2.py)
- [`3-08_code-for-exercise-3-4.py`](3-08_code-for-exercise-3-4.py)
- [`3-09_loading-travel-data-in-problem-3-1.py`](3-09_loading-travel-data-in-problem-3-1.py) — downloads data

### Chapter 4 — Random coefficients and demand estimation

- [`4-01_laguerre-diagrams.py`](4-01_laguerre-diagrams.py) — requires `sdot`
- [`4-02_ar-ghk-simulators.py`](4-02_ar-ghk-simulators.py)
- [`4-03_probit-market-share.py`](4-03_probit-market-share.py)
- [`4-04_demand-estimation-with-simple-iv.py`](4-04_demand-estimation-with-simple-iv.py)
- [`4-05_demand-estimation-with-iv-gmm-logit.py`](4-05_demand-estimation-with-iv-gmm-logit.py)
- [`4-06_demand-estimation-with-iv-gmm-rcl.py`](4-06_demand-estimation-with-iv-gmm-rcl.py)
- [`4-07_the-blp-method.py`](4-07_the-blp-method.py)
- [`4.08_loading-blp-auto-data.py`](4.08_loading-blp-auto-data.py) — downloads data

### Chapter 5 — Optimal transport, gravity, matching, and coalitions

- [`5-01_ot-via-glm.py`](5-01_ot-via-glm.py)
- [`5-02_ot-via-ipfp.py`](5-02_ot-via-ipfp.py)
- [`5-03_gravity-via-glm.py`](5-03_gravity-via-glm.py)
- [`5-04_gravity-via-sista.py`](5-04_gravity-via-sista.py)
- [`5-05_cupids-via-glm.py`](5-05_cupids-via-glm.py)
- [`5-06_coalition-via-glm.py`](5-06_coalition-via-glm.py)
- [`5-07_coalition-via-glm.py`](5-07_coalition-via-glm.py)
- [`5-08_loading-choo-and-siow-data.py`](5-08_loading-choo-and-siow-data.py) — downloads data
- [`5-09_loading-trade-data.py`](5-09_loading-trade-data.py) — downloads data
- [`5-10_code-problem-eaton-kortum.py`](5-10_code-problem-eaton-kortum.py)

### Chapter 6 — Dynamic discrete choice

- [`6-01_ddc-nohet-finite-lp.py`](6-01_ddc-nohet-finite-lp.py) — requires Gurobi
- [`6-02_ddc-nohet-finite-bwd-fwd-induc.py`](6-02_ddc-nohet-finite-bwd-fwd-induc.py) — requires Gurobi
- [`6-03_ddc-logit-finite-estimation-numpy.py`](6-03_ddc-logit-finite-estimation-numpy.py)
- [`6-04_ddc-logit-finite-estimation-torch.py`](6-04_ddc-logit-finite-estimation-torch.py) — requires PyTorch
- [`6-05_ddc-logit-infinite-estimation-nfpx.py`](6-05_ddc-logit-infinite-estimation-nfpx.py)
- [`6-06_ddc-logit-infinite-estimation-augmented-lagrangian.py`](6-06_ddc-logit-infinite-estimation-augmented-lagrangian.py)
- [`6-07_ccp-estimator-ddc.py`](6-07_ccp-estimator-ddc.py)
- [`6-08_pb-rust-data.py`](6-08_pb-rust-data.py) — uses Rust bus-engine data
- [`6-09_dynamic-matching-estimation.py`](6-09_dynamic-matching-estimation.py)

### Chapter 7 — Constrained choice and allocation

- [`7-01_constrained-logit.py`](7-01_constrained-logit.py)
- [`7-02_code-simulation-constrained-choice.py`](7-02_code-simulation-constrained-choice.py) — requires Gurobi
- [`7-03_code-classes-of-constrained-choice.py`](7-03_code-classes-of-constrained-choice.py) — requires Gurobi
- [`7-04_code-deferred-acceptance.py`](7-04_code-deferred-acceptance.py) — requires Gurobi

### Chapter 8 — Matching models

- [`8-01_matching-models-general.py`](8-01_matching-models-general.py)
- [`8-02_matching-models-probit.py`](8-02_matching-models-probit.py) — requires Gurobi
- [`8-03_ddc-estimation-simulation-finite.py`](8-03_ddc-estimation-simulation-finite.py) — requires Gurobi

## Checking syntax

To check all Python files for syntax errors without running the examples:

```bash
python -m compileall code
```

This does not verify numerical correctness, package compatibility, data availability, solver licences, or runtime requirements.

## Reporting issues

Bug reports, corrections, and suggestions are welcome through the repository's [Issues](https://github.com/alfredgalichon/DCM/issues) page. When reporting a problem, please include:

- the filename;
- your operating system;
- your Python version;
- the relevant package versions;
- the complete error message.

## Licence

Unless otherwise indicated in an individual file, the original source code in this directory is distributed under the GNU General Public License v3.0. See [`LICENSE.txt`](LICENSE.txt).
