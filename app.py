
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 House Price Prediction")
st.write(
    "Predict the estimated median house value using "
    "Linear Regression and the California Housing dataset."
)

st.info(
    "Dataset: California Housing from Scikit-learn. "
    "Predictions are estimates, not actual market valuations."
)

# --------------------------------------------------
# Load dataset and train model
# --------------------------------------------------
@st.cache_resource
def load_and_train():
    housing = fetch_california_housing(as_frame=True)

    df = housing.frame.copy()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Split before fitting preprocessing to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Imputation and normalization are learned only from training data
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return (
        df, X, y, X_train, X_test, y_test,
        y_pred, model, mse, rmse, mae, r2
    )


try:
    (
        df, X, y, X_train, X_test, y_test,
        y_pred, model, mse, rmse, mae, r2
    ) = load_and_train()

except Exception as e:
    st.error("Could not load the dataset or train the model.")
    st.write(
        "Check your internet connection. Scikit-learn may need "
        "to download the dataset the first time the application runs."
    )
    st.exception(e)
    st.stop()

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Home & Prediction",
        "Dataset Exploration",
        "Model Evaluation"
    ]
)

# --------------------------------------------------
# Home and prediction
# --------------------------------------------------
if page == "Home & Prediction":

    st.subheader("Enter House Details")

    st.write(
        "Adjust the property features below to estimate "
        "the median house value."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            med_income = st.number_input(
                "Median Income (in tens of thousands of dollars)",
                min_value=0.5,
                max_value=15.0,
                value=5.0,
                step=0.1
            )

            house_age = st.number_input(
                "Median House Age (years)",
                min_value=1.0,
                max_value=52.0,
                value=20.0,
                step=1.0
            )

            avg_rooms = st.number_input(
                "Average Rooms per Household",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.1
            )

            avg_bedrooms = st.number_input(
                "Average Bedrooms per Household",
                min_value=0.2,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        with col2:
            population = st.number_input(
                "Block Population",
                min_value=1.0,
                max_value=40000.0,
                value=1000.0,
                step=100.0
            )

            avg_occupancy = st.number_input(
                "Average Occupancy",
                min_value=0.5,
                max_value=20.0,
                value=3.0,
                step=0.1
            )

            latitude = st.number_input(
                "Latitude (California)",
                min_value=32.0,
                max_value=42.0,
                value=36.7,
                step=0.1
            )

            longitude = st.number_input(
                "Longitude (California)",
                min_value=-125.0,
                max_value=-114.0,
                value=-119.5,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Predict House Price",
            use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame([{
            "MedInc": med_income,
            "HouseAge": house_age,
            "AveRooms": avg_rooms,
            "AveBedrms": avg_bedrooms,
            "Population": population,
            "AveOccup": avg_occupancy,
            "Latitude": latitude,
            "Longitude": longitude
        }])

        predicted_value = float(model.predict(input_data)[0])

        # Dataset target is expressed in units of $100,000.
        predicted_price = predicted_value * 100000

        st.divider()
        st.subheader("Estimated House Price")

        if predicted_value < 0:
            st.warning(
                "The model produced a negative estimate. "
                "This indicates a limitation of Linear Regression "
                "for these inputs; it is not a valid house price."
            )
        else:
            st.success(
                f"Estimated median value: ${predicted_price:,.0f}"
            )

        st.caption(
            "This is a model estimate based on California census "
            "block-group data, not an individual property's appraisal."
        )

# --------------------------------------------------
# Dataset exploration
# --------------------------------------------------
elif page == "Dataset Exploration":

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Number of Features", X.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("### Feature Distributions")

    feature = st.selectbox(
        "Choose a feature",
        df.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Feature Correlations")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# Model evaluation
# --------------------------------------------------
elif page == "Model Evaluation":

    st.subheader("Linear Regression Performance")

    st.write(
        f"Training samples: {len(X_train):,} | "
        f"Testing samples: {len(X_test):,}"
    )

    # The dataset target is measured in units of $100,000.
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MSE", f"{mse:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("MAE", f"{mae:.4f}")
    col4.metric("R² Score", f"{r2:.4f}")

    st.caption(
        "MSE, RMSE, and MAE are in dataset target units. "
        "Multiply RMSE or MAE by $100,000 for approximate dollar "
        "errors. MSE is in squared target units."
    )

    st.markdown("### Actual vs Predicted Prices")

    comparison = pd.DataFrame({
        "Actual ($100,000 units)": y_test.to_numpy(),
        "Predicted ($100,000 units)": y_pred
    })

    st.dataframe(
        comparison.head(20).round(3),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.4)

    lower = min(float(y_test.min()), float(y_pred.min()))
    upper = max(float(y_test.max()), float(y_pred.max()))

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--"
    )

    ax.set_xlabel("Actual House Value ($100,000 units)")
    ax.set_ylabel("Predicted House Value ($100,000 units)")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Model Interpretation")

    st.write(
        f"- **MSE:** {mse:.4f} — average squared prediction error.\n"
        f"- **RMSE:** {rmse:.4f} — prediction error in target units.\n"
        f"- **MAE:** {mae:.4f} — average absolute prediction error.\n"
        f"- **R²:** {r2:.4f} — proportion of test-set target variance "
        "explained by the model."
    )

    st.markdown("### Feature Coefficients")

    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.named_steps["regressor"].coef_
    }).sort_values(
        by="Coefficient",
        key=abs,
        ascending=False
    )

    st.dataframe(
        coefficients.round(4),
        use_container_width=True
    )

    st.caption(
        "Coefficients correspond to standardized input features. "
        "Their signs indicate the direction of each feature's "
        "linear association, holding other features constant."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Built with Python, Streamlit, Pandas, Matplotlib, "
    "Seaborn and Scikit-learn."
)