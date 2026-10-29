# Prediction and Ranking of Companies

**Master's project, Mila / Université de Montréal (2020), in partnership with CDPQ (Caisse de dépôt et placement du Québec).**

Can machine learning surface the companies most likely to outperform their sector over the next 12 months, in a way that investors can understand and trust?

This project framed that question three ways (regression, classification and learning-to-rank), compared classical time-series, tree-based and deep-learning models under a leakage-safe evaluation, and backtested the most useful one as a stock-selection tool.

## Headline result

An **XGBoost learning-to-rank model** ranked every company in the universe each month. Equal-weighting the model's **top 20** picks every month over the held-out test period (2015 to 2019) produced an **average 12-month excess return of +8.17%** versus investing in the whole universe.

| Portfolio (monthly, equal-weighted) | Avg. 12-month excess return |
|---|---|
| Top 20 | **+8.17%** |
| Top 50 | +6.50% |
| Top 100 | +6.55% |
| Top 150 | +5.61% |

- Smaller portfolios earned more on average but were more volatile: the top 20 ranged from above +60% (May 2017 entry) to about −40% (June 2019 entry). Widening to the top 150 cut the June 2019 drawdown to a positive +5%, at the cost of about 1 point of average return.
- **Sanity check against luck:** randomly sampling 20 to 100 names from the model's top 150 each month still gave positive average excess returns in every configuration (e.g. +4.85% with 70 names), suggesting the edge comes from the ranking rather than from portfolio size or luck.
- **Caveats:** no transaction costs or fees, equal weighting of every pick, and the ranker used XGBoost's default hyperparameters (no tuning yet).

## Problem

CDPQ's investment team wanted to know whether its internal data on one sector could highlight the best (and worst) companies to support investment decisions. Because the end users are investors rather than ML specialists, **interpretability was a hard requirement**: every model was judged on the trade-off between performance and how easily its decisions can be explained.

## Data (high level)

The data is proprietary to CDPQ and is not included in this repository. The sector, companies and most feature names are confidential.

- **Universe:** 2,063 companies in a single sector, each tagged with a sub-sector.
- **History:** 148,298 daily rows from 1994 to 2020, merged from several data providers (so some features are missing for some companies and years).
- **Features:** index membership (S&P 400 / 500 / 600), trailing 3, 6, 12, 24 and 36-month returns, sub-sector, and other confidential fundamentals.
- **Target:** forward 12-month return, and mostly the **12-month excess return** (a company's return minus the universe average), since the goal is to find outperformers.
- Companies enter and leave the universe over time (listings, sector reclassification, bankruptcies).

## Leakage-safe evaluation

Financial time series make it easy to leak the future into training. Two safeguards were used throughout:

1. **Monthly resampling.** Daily values barely change day to day, so data was reduced to monthly snapshots to remove redundancy.
2. **Purged chronological splits** (following López de Prado, *Advances in Financial Machine Learning*). Because the target looks 12 months ahead, a full one-year gap was removed between each split so no label overlaps the next set:

| Split | Years |
|---|---|
| Train | 1994 to 2009 |
| *Purged* | *2010* |
| Validation | 2011 to 2013 |
| *Purged* | *2014* |
| Test | 2015 to 2019 |

Hyperparameters were selected on the validation set. Clustering for the clustered LSTMs used features only, never targets, to avoid look-ahead bias. An expanding-window scheme was also prototyped but dropped as too much overhead for the gain.

## Methods

| Model | Used for | Notes |
|---|---|---|
| **ARIMAX** | Regression | One model per company (1,021 companies with at least 36 months of data), grid search over (p, d, q) |
| **Gradient boosting (XGBoost)** | Regression, classification | One model for the whole universe; random hyperparameter search; feature importance, tree inspection and SHAP for explanations |
| **XGBoost with time-series windows** | Regression, classification | Previous 5 months of features and returns, flattened |
| **LSTM (Keras)** | Regression, classification | 5-month windows kept as sequences; dropout, Adam, early stopping; per-timestep SHAP |
| **LSTM with clustering (tslearn)** | Regression, classification | Companies clustered into 3 groups, one LSTM per cluster |
| **XGBoost learning-to-rank** | Ranking | Pairwise ranking loss, companies grouped by month |

Built on AWS (SageMaker for training and notebooks, S3 for data), with scikit-learn, XGBoost, Keras, tslearn and SHAP.

## Results by task

**Regression (12-month excess return, test MSE).** Baseline: mean of the past 5 months, MSE 0.6649.

- XGBoost (single month of features) was the best regression model, beating the baseline by about 13% with a shallow, readable model (50 depth-1 trees).
- Adding 5-month windows to XGBoost overfit (test MSE 0.673, worse than baseline).
- A single LSTM matched the baseline (0.6682); clustering plus one LSTM per cluster improved it to a weighted 0.6519, at the cost of three separate trainings and searches.
- ARIMAX tracked short horizons reasonably but diverged over longer horizons, needs one model per company, and cannot score companies with short histories, so it was impractical for this use.

**Classification (will excess return be positive?).** Baseline: majority class, 58.40% accuracy.

- XGBoost reached 58.90% test accuracy and identified about 60% of underperformers; XGBoost with windows 59.40%; LSTM 59.59%.
- The gains over baseline were small and did not justify the extra complexity of the windowed and LSTM models. Clustered LSTMs did worst (51 to 53%).

**Ranking.** The most useful framing for investors, and the best result (see the headline above). A ranker sits between a fully automated strategy and a discretionary process: it shortlists companies each month and leaves the final call to the investor.

## Takeaways

- Framing matters more than model choice here: turning the problem into a monthly ranking gave investors something actionable, where predicting the exact return did not.
- Simple, explainable models won. XGBoost beat the LSTMs on regression, was within 0.2 points on classification, trained much faster and could be explained with feature importance and SHAP.
- Honest evaluation is most of the work in financial ML: monthly resampling, purged splits and target-free clustering were what made the results trustworthy.

## Suggested next steps (from the report)

- A neural ranker (e.g. LSTM with triplet loss) to compare against XGBoost ranking.
- Backtest the regression and classification models the same way as the ranker.
- Combine tasks: keep only ranker picks that the classifier also labels as outperformers and whose predicted return clears a threshold.
- Add transaction costs and realistic position sizing to the simulation.

## About this repository

This repository documents the project. The code and data belong to CDPQ and are not published here.

**Author:** Ilyas Amaniss, Master's in Machine Learning, Mila / Université de Montréal
