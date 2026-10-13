# House Price Prediction using Streamlit

A beginner-friendly machine learning project that predicts estimated median house values using **Linear Regression** and an interactive Streamlit web application.

## Features

- California Housing dataset from Scikit-learn
- Dataset preview and statistical summaries
- Feature distribution visualizations
- Missing-value handling using median imputation
- Numerical feature normalization using StandardScaler
- Training and testing data split
- Linear Regression model
- House price prediction through an interactive interface
- Model evaluation using MSE, RMSE, MAE, and R²
- Actual vs. predicted value visualization
- Feature correlation heatmap and regression coefficients

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure

```text
House-Price-Prediction/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`, to open the application in your browser.

An internet connection may be required the first time the California Housing dataset is downloaded.

## Machine Learning Workflow

1. Load and explore the dataset.
2. Separate input features and the target variable.
3. Split the data into training and testing sets.
4. Impute missing values and standardize the features.
5. Train a Linear Regression model.
6. Generate predictions and evaluate performance.
7. Use the interactive interface to estimate house values.

## Evaluation Metrics

- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R²:** Coefficient of determination

## Dataset

The California Housing dataset contains census block-group data. Its target is median house value, expressed in units of $100,000.

The predictions are educational estimates and should not be treated as real estate appraisals.

## Learning Outcomes

- Tabular data preprocessing
- Exploratory data analysis
- Feature scaling
- Regression modeling
- Model evaluation
- Interactive machine learning applications with Streamlit