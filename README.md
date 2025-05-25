Airbnb Price Prediction ML Pipeline
Description

This repository contains a Python script (scripts/airbnb_price_pipeline.py) for exploratory data analysis, cleaning, and price prediction for Airbnb rentals. Starting from a raw dataset (data/listings.csv), the script performs:

    Data loading and cleaning (removal of irrelevant columns, handling dates and missing values)
    Exploration (descriptive statistics, visualizations with Seaborn/Matplotlib)
    Feature Engineering:
        Label Encoding of categorical variables (e.g., neighbourhood_group, room_type)
        Creation of derived variables (price_range, IQR filters, etc.)
    Data Preparation for ML:
        Log-transform of the target (price → log10(price))
        Train/test split (25% test)
        Selective scaling with ColumnTransformer (float columns only)
    Model Training and Evaluation:
        Linear models on scaled data: Linear Regression, Bayesian Ridge
        Tree-based models on unscaled data: Decision Tree Regressor, Gradient Boosting Regressor
        Performance metrics: RMSE, R², MAE
        Cross-validation (5-fold) for Gradient Boosting
    Final Visualizations:
        Correlation matrix post-encoding
        Actual vs. Predicted plot for each model
        Outlier analysis (residuals boxplot, IQR removal)

Repository Structure
Bash

    data/                     # Raw and processed datasets
     ├── listings.csv          # Original Airbnb dataset
    scripts/                  # Analysis and modeling scripts
     └── airbnb_price_pipeline.py
    notebooks/                # Interactive analysis (EDA)
     └── eda_airbnb.ipynb
    tests/                    # (optional) Unit tests
    README.md                 # Project documentation
    requirements.txt          # Python dependencies

Requirements

    Python 3.8+
    Python packages (see requirements.txt)

Key Dependencies
Bash

numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
xgboost
scipy
folium (optional)

Installation

    Clone the repository:
    Bash

    git clone https://github.com/JacobHess03/ML-AirBNB-price-prediction/tree/main
    cd ML-AirBNB-price-prediction

Create and activate a virtual environment:
Bash

    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows

Install dependencies:
Bash

    pip install -r requirements.txt

Running the Script
Bash

    python scripts/airbnb_price_pipeline.py

This command sequentially executes:

    Data cleaning and transformations
    Exploratory data analysis and plotting
    ML data preparation (split, scaling)
    Training and evaluating 4 models
    Visualization of results and outlier management

Customizations

    Dataset Path: Modify data/listings.csv in airbnb_price_pipeline.py if needed.
    Model Parameters: Adjust max_depth, n_estimators, learning rate, etc.
    Feature Selection: Update the X_cols list to include or exclude variables.
    Outlier Filters: Change IQR strategy or thresholds in the code.

Expected Results

    EDA plots (histograms, box plots, scatter plots, maps)
    Annotated correlation matrix
    Comparison table of RMSE/R²/MAE for each model
    Actual vs. Predicted plots and residuals
    Report on removed outliers and R² improvement

Contributions

Contributions, pull requests, and issues are welcome! Please make sure to open an issue to discuss substantial changes.
License

MIT License.

Author: Giacomo Visciotti
