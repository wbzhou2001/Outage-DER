# Impact of power outages on the adoption of residential solar photovoltaic in a changing climate

[![arXiv](https://img.shields.io/badge/arXiv-2512.05027-b31b1b.svg)](https://arxiv.org/abs/2512.05027)

Spatio-temporal modeling of power outage events across substations using weather covariates and grid topology.
The repository implements a full pipeline from data preprocessing to forecasting and visualization.

---

## Overview

This project builds a data-driven framework for modeling outage dynamics and forecasting resilience metrics.
The core workflow is implemented in `plotting.ipynb`.

Main components:

* Data cleaning and alignment
* Spatio-temporal covariate construction
* Temporal data splitting (strictly time-based)
* Multivariate Hawkes process modeling
* Sequential simulation for evaluation and forecasting
* Resilience metric computation (SAIFI, SAIDI, CAIDI)
* Visualization (temporal and spatial)

---

## Repository Structure

```
data/        # raw and processed datasets
model/       # Hawkes process implementation
utils/       # helper functions and configuration
cache/       # intermediate results
plotting.ipynb
README.md
```

---

## Data Processing

### Substation Alignment

* Load `topology.csv` to define valid substations
* Remove invalid entries (e.g., `"nan"`)
* Ensure consistent ordering across:

  * outage events
  * covariates
  * model inputs

---

### Weather Covariates

* Source: `7_1_2_cov_weather.csv`

* Convert from long format to tensor:

  * axis 1: substations
  * axis 2: monthly timestamps
  * axis 3: weather features

* Each feature is constructed via pivot:

  * `substation × month`

* Final shape:

  `(n_substations, n_months, n_features)`

---

### Covariate Normalization

* Apply MinMax scaling across all entries
* Flatten → scale → reshape

This ensures stable optimization and comparable feature magnitudes.

---

### Outage Events

* Source: `8_20_outage_by_substation.csv`

* Convert `Time` to datetime

* Filter events to:

  **2014-01 < Time < 2024-01**

* Map substations to integer indices

* Remove unmatched records

---

### Event Transformation

Each event is represented as:

* normalized time
* scaled customer interruptions (CI)
* scaled outage duration
* substation index

Processing includes:

* time normalization to `[0, 1]`
* log-transform of CI and duration
* MinMax scaling

---

## Data Splitting

All splits are strictly chronological.

### Key timestamps

* Start time: **2014-01**
* Calibration split: **2015-01**
* Prediction start: **2024-01**
* End time: **2040-01**

---

### Event Data

* Training set:
  `Time < 2015-01`

* Calibration / evaluation set:
  `Time ≥ 2015-01`

So:

* Training period: **2014-01 – 2014-12**
* Evaluation period: **2015-01 – 2023-12**

---

### Covariates

* Training covariates:
  months `< 2015-01`

* Testing covariates:
  months `≥ 2015-01`

So:

* Training covariates: **2014-01 – 2014-12**
* Testing covariates: **2015-01 – 2040-01**

---

### Backtesting

* Window: **2015-01 – 2023-12**
* Used for:

  * held-out evaluation
  * sequential simulation

---

### Forecasting

* Window: **2024-01 – 2039-12**
* Used for:

  * future prediction
  * scenario analysis

---

### Key Principle

No future information is used in training.
All evaluation and forecasting follow strict temporal causality.

---

## Modeling

### Hawkes Process

* Multivariate Hawkes model
* Each substation is a dimension
* Captures:

  * temporal clustering of outages
  * cross-substation interactions

---

### Sequential Simulation

* Simulate events month-by-month
* Update history after each step
* Continue forward in time

This mirrors real-world deployment.

---

## Outputs

* `data/8_28_pred_results.csv`

Contains:

* SAIFI
* SAIDI
* CAIDI
* prediction intervals

---

## Visualization

The notebook includes:

* Temporal plots with prediction intervals
* Spatial plots using substation polygons
* Clear separation between historical and forecast periods

---

## Usage

```bash
git clone https://github.com/wbzhou2001/Outage-DER.git
cd Outage-DER
jupyter notebook plotting.ipynb
```
