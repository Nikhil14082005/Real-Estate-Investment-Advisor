import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Real Estate Investment Advisor")
st.write(
    "Predict a property's estimated value after five years, calculate "
    "estimated appreciation, and compare investment options."
)

# ---------------------------------------------------------
# LOAD TRAINED MODELS
# Keep all three .joblib files in the same folder as this app.py
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    regression_model = joblib.load("best_regression_model.joblib")
    classification_model = joblib.load("best_classification_model.joblib")
    model_config = joblib.load("model_config.joblib")
    return regression_model, classification_model, model_config


try:
    reg_model, cls_model, config = load_models()
    features = config["features"]
    threshold = float(config["appreciation_threshold"])
except Exception as error:
    st.error(
        "Could not load the model files. Make sure these files are in the "
        "same folder as app.py: best_regression_model.joblib, "
        "best_classification_model.joblib, and model_config.joblib."
    )
    st.exception(error)
    st.stop()

st.sidebar.header("Investment Rule")
st.sidebar.metric("Good investment threshold", f"{threshold:.1f}%")
st.sidebar.caption(
    "The final recommendation is YES when predicted five-year appreciation "
    "is equal to or greater than this threshold."
)

# ---------------------------------------------------------
# COMMON FUNCTIONS
# ---------------------------------------------------------
def create_input_row(values):
    """Create model input in the exact feature order saved in model_config."""
    return pd.DataFrame([values])[features]


def analyze_property(values):
    """Run both models and calculate appreciation/profit."""
    row = create_input_row(values)

    future_price = float(reg_model.predict(row)[0])
    current_price = float(values["Current_Price"])
    estimated_profit = future_price - current_price
    appreciation = (estimated_profit / current_price) * 100

    # Final recommendation follows the configured appreciation threshold.
    good_investment = appreciation >= threshold

    # Keep the classifier prediction as a separate ML output for reference.
    classifier_prediction = cls_model.predict(row)[0]

    classifier_yes_probability = None
    if hasattr(cls_model, "predict_proba"):
        probabilities = cls_model.predict_proba(row)[0]
        class_labels = list(cls_model.classes_)
        if 1 in class_labels:
            classifier_yes_probability = float(
                probabilities[class_labels.index(1)]
            )

    return {
        "Current Price": current_price,
        "Predicted 5-Year Price": future_price,
        "Estimated Profit/Loss": estimated_profit,
        "Appreciation (%)": appreciation,
        "Good Investment": "YES" if good_investment else "NO",
        "Classifier Model Output": str(classifier_prediction),
        "Classifier YES Probability": classifier_yes_probability,
    }


def property_inputs(prefix, defaults=None):
    """Display property input fields and return their values."""
    defaults = defaults or {
        "Area_sqft": 3000,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Floors": 2,
        "Year_Built": 2015,
        "Location_Score": 80,
        "School_Score": 75,
        "Current_Price": 7000000.0,
    }

    st.markdown(f"**{prefix}**")
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            f"{prefix} - Area (sqft)", min_value=100, max_value=50000,
            value=int(defaults["Area_sqft"]), step=100, key=f"{prefix}_area"
        )
        bedrooms = st.number_input(
            f"{prefix} - Bedrooms", min_value=0, max_value=20,
            value=int(defaults["Bedrooms"]), key=f"{prefix}_bedrooms"
        )
        bathrooms = st.number_input(
            f"{prefix} - Bathrooms", min_value=0, max_value=20,
            value=int(defaults["Bathrooms"]), key=f"{prefix}_bathrooms"
        )
        floors = st.number_input(
            f"{prefix} - Floors", min_value=0, max_value=100,
            value=int(defaults["Floors"]), key=f"{prefix}_floors"
        )

    with col2:
        year_built = st.number_input(
            f"{prefix} - Year Built", min_value=1800, max_value=2100,
            value=int(defaults["Year_Built"]), key=f"{prefix}_year"
        )
        location_score = st.slider(
            f"{prefix} - Location Score", min_value=0, max_value=100,
            value=int(defaults["Location_Score"]), key=f"{prefix}_location"
        )
        school_score = st.slider(
            f"{prefix} - School Score", min_value=0, max_value=100,
            value=int(defaults["School_Score"]), key=f"{prefix}_school"
        )
        current_price = st.number_input(
            f"{prefix} - Current Price (₹)", min_value=1.0,
            value=float(defaults["Current_Price"]), step=10000.0,
            key=f"{prefix}_price"
        )

    return {
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Year_Built": year_built,
        "Location_Score": location_score,
        "School_Score": school_score,
        "Current_Price": current_price,
    }


def show_result(result):
    """Display one property's prediction results."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Price After 5 Years", f"₹{result['Predicted 5-Year Price']:,.2f}")
    c2.metric("Estimated Profit / Loss", f"₹{result['Estimated Profit/Loss']:,.2f}")
    c3.metric("5-Year Appreciation", f"{result['Appreciation (%)']:.2f}%")

    if result["Good Investment"] == "YES":
        st.success(f"Good Investment: YES — appreciation meets the {threshold:.1f}% rule.")
    else:
        st.warning(f"Good Investment: NO — appreciation is below the {threshold:.1f}% rule.")

    st.caption(
        f"Separate classification-model output: {result['Classifier Model Output']}"
    )
    if result["Classifier YES Probability"] is not None:
        st.write(
            "Classification model probability for class 1 (YES): "
            f"{result['Classifier YES Probability']:.1%}"
        )
        st.progress(result["Classifier YES Probability"])


# ---------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------
prediction_tab, comparison_tab, information_tab = st.tabs(
    ["🔍 Property Prediction", "📊 Compare Properties", "ℹ️ Model Information"]
)

# ---------------------------------------------------------
# TAB 1: SINGLE PROPERTY PREDICTION
# ---------------------------------------------------------
with prediction_tab:
    st.subheader("Enter Property Details")

    with st.form("single_property_form"):
        property_data = property_inputs("Property")
        predict_button = st.form_submit_button("Analyze Property")

    if predict_button:
        result = analyze_property(property_data)
        st.subheader("Prediction Results")
        show_result(result)

        chart_data = pd.DataFrame({
            "Price": [
                result["Current Price"],
                result["Predicted 5-Year Price"]
            ]
        }, index=["Current Price", "Predicted Price After 5 Years"])

        st.subheader("Current vs Predicted Property Value")
        st.bar_chart(chart_data)

        st.subheader("Investment Summary")
        st.write(
            f"The estimated five-year value is ₹{result['Predicted 5-Year Price']:,.2f}. "
            f"The estimated change in value is ₹{result['Estimated Profit/Loss']:,.2f} "
            f"({result['Appreciation (%)']:.2f}%)."
        )

# ---------------------------------------------------------
# TAB 2: COMPARE UP TO THREE PROPERTIES
# ---------------------------------------------------------
with comparison_tab:
    st.subheader("Compare Up to Three Properties")
    st.write("Enter details for each property and compare their model predictions.")

    with st.form("comparison_form"):
        compare_col1, compare_col2, compare_col3 = st.columns(3)

        with compare_col1:
            property_1 = property_inputs("Property 1")
        with compare_col2:
            property_2 = property_inputs(
                "Property 2",
                {
                    "Area_sqft": 1500, "Bedrooms": 2, "Bathrooms": 2,
                    "Floors": 1, "Year_Built": 2015, "Location_Score": 60,
                    "School_Score": 65, "Current_Price": 4500000.0
                }
            )
        with compare_col3:
            property_3 = property_inputs(
                "Property 3",
                {
                    "Area_sqft": 900, "Bedrooms": 1, "Bathrooms": 1,
                    "Floors": 1, "Year_Built": 2005, "Location_Score": 30,
                    "School_Score": 35, "Current_Price": 3500000.0
                }
            )

        compare_button = st.form_submit_button("Compare Properties")

    if compare_button:
        results = []
        for name, data in [
            ("Property 1", property_1),
            ("Property 2", property_2),
            ("Property 3", property_3),
        ]:
            item = analyze_property(data)
            item["Property"] = name
            results.append(item)

        comparison_df = pd.DataFrame(results).set_index("Property")

        st.subheader("Comparison Results")
        display_columns = [
            "Current Price",
            "Predicted 5-Year Price",
            "Estimated Profit/Loss",
            "Appreciation (%)",
            "Good Investment",
        ]
        st.dataframe(comparison_df[display_columns].style.format({
            "Current Price": "₹{:,.2f}",
            "Predicted 5-Year Price": "₹{:,.2f}",
            "Estimated Profit/Loss": "₹{:,.2f}",
            "Appreciation (%)": "{:.2f}%",
        }), use_container_width=True)

        st.subheader("Current Price vs Predicted Five-Year Price")
        price_chart = comparison_df[
            ["Current Price", "Predicted 5-Year Price"]
        ]
        st.bar_chart(price_chart)

        st.subheader("Five-Year Appreciation Comparison")
        appreciation_chart = comparison_df[["Appreciation (%)"]]
        st.bar_chart(appreciation_chart)

        best_property = comparison_df["Predicted 5-Year Price"].idxmax()
        st.info(
            f"{best_property} has the highest predicted five-year property value. "
            "This does not automatically mean it has the highest return percentage "
            "or is the safest investment."
        )

        csv_data = comparison_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Comparison as CSV",
            data=csv_data,
            file_name="property_comparison.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------
# TAB 3: MODEL INFORMATION
# ---------------------------------------------------------
with information_tab:
    st.subheader("How This Application Works")
    st.markdown(
        """
        **Regression model**
        - Predicts the estimated property price after five years.
        - Estimated profit/loss = predicted five-year price − current price.
        - Appreciation = (estimated profit/loss ÷ current price) × 100.

        **Classification model**
        - Produces a separate Good Investment class prediction.
        - Its output and probability are shown for reference.

        **Final dashboard recommendation**
        - YES when regression-based five-year appreciation is at least the
          configured threshold.
        - NO when appreciation is below the configured threshold.
        """
    )
    st.write("Features used by the models:")
    st.write(", ".join(features))
    st.caption(
        "The threshold is a project setting, not a universal real-estate rule. "
        "Predictions are estimates based on the training dataset and are not financial advice."
    )
