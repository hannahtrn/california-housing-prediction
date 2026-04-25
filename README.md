# California Housing Price Prediction

Team machine learning project predicting California single-family home sale prices, completed during my Data Science internship at IDXExchange (Jan 2026 – Present) as part of team DS38.

This repository documents the project's methodology, modeling decisions, and results. The original code is not public as it was developed under IDXExchange.

## Problem

Predict sale prices for California single-family residential homes using historical MLS records. The dataset spanned June 2025 to February 2026, with rolling-forward validation across multiple time windows to test model stability over time.

- **Training:** June – November 2025
- **Validation:** December 2025
- **Testing:** January & February 2026

## Approach

### Feature Engineering

Started with the raw feature set and iterated heavily:

- **Removed** features that added noise or were poorly populated (`StreetNumberNumeric`, `AssociationFee`, `LotSizeAcres`)
- **Added** `Latitude`, `Longitude`, `PostalCode`, and `LotSizeSquareFeet`
- **Log-transformed** `ClosePrice`, `LivingArea`, and `LotSizeSquareFeet` to reduce skewness and stabilize variance
- **Engineered spatial features** (`Latitude²`, `Longitude²`, `Latitude × Longitude`) to capture nonlinear geographic price patterns
- **Encoded categoricals** (`ViewYN`, `PoolPrivateYN`, `FirePlaceYN`, `NewConstructionYN`, `PostalCode`) as dummy variables
- **Kept structural features linear** (Bedrooms, Bathrooms, Parking, GarageSpaces, Stories, YearBuilt)

### Modeling

Compared multiple model families against a log-linear regression baseline:

| Model | MdAPE | R² |
|---|---|---|
| Ridge Regression | 38.27% | 0.472 |
| Log-linear Regression (baseline) | 22.80% | 0.670 |
| Decision Tree | 15.41% | -4.94 |
| Random Forest | 9.01% | -3.32 |
| **XGBoost** | **8.00%** | **0.781** |

Tree-based models without proper regularization showed strong MdAPE but negative R² — clear overfitting and poor generalization. Boosted models handled the skewed, heterogeneous feature space far better.

After tuning, focused on three final candidates:

| Model | MdAPE (Dec) | R² (Dec) |
|---|---|---|
| XGBoost | 7.62% | 0.875 |
| LightGBM | 7.74% | 0.883 |
| **Ensemble (Stacked XGBoost + LightGBM)** | **7.74%** | **0.880** |

### Rolling-Forward Holdout

Tested temporal stability across December, January, and February:

| Model | Dec MdAPE | Jan MdAPE | Feb MdAPE | Drift |
|---|---|---|---|---|
| XGBoost | 7.62% | 7.77% | 7.80% | 0.18% |
| LightGBM | 7.74% | 7.78% | 8.12% | 0.38% |
| **Ensemble** | **7.74%** | **7.74%** | **7.88%** | **0.14%** |

The stacked ensemble had the lowest temporal drift, making it the most reliable model for ongoing prediction.

## Key Takeaways

- **XGBoost alone had the lowest absolute MdAPE**, but the ensemble was more robust over time
- **Spatial polynomial features** (lat², lon², lat × lon) meaningfully improved model performance by letting linear models capture geographic nonlinearity
- **R² and MdAPE can disagree sharply** — Decision Trees and Random Forests had decent MdAPE but negative R², a clear signal of overfitting that pure error metrics would have missed

## Future Directions

- Segment data by price range and train range-specific models
- More sophisticated imputation for missing values
- Explore deeper nonlinear transformations and additional spatial features

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn, Jupyter Notebook

## What I Learned

- The full ML loop end-to-end: messy raw data in, validated production-quality model out
- That feature engineering and preprocessing decisions often matter more than model choice
- How to evaluate models across multiple metrics, not just one, and why temporal stability matters when deploying
- How to translate model results into recommendations a non-technical audience can act on

---

*Team project at IDXExchange (DS38: Tony Bai, Sonia Alimchandani, Nathan Oliver, Hannah Tran, Cathy Fang). Code is proprietary; this README documents methodology and outcomes.*
