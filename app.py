
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model_bundle = joblib.load("best_model.pkl")

if isinstance(model_bundle, dict):
    model = model_bundle["pipeline"]
    features = model_bundle["features"]
else:
    model = model_bundle
    features = list(model.feature_names_in_)

print("Model loaded successfully!")
print("Features:", features)


# Dropdown options
DROPDOWN_OPTIONS = {
    "Weather": ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy"],
    "Traffic_Level": ["Low", "Medium", "High"],
    "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"],
    "Vehicle_Type": ["Bike", "Scooter", "Car"]
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    input_values = {}

    # Defaults shown when the page first opens
    defaults = {
        "Distance_km": "5",
        "Weather": "Clear",
        "Traffic_Level": "Medium",
        "Time_of_Day": "Afternoon",
        "Vehicle_Type": "Bike",
        "Preparation_Time_min": "15",
        "Courier_Experience_yrs": "2"
    }

    if request.method == "POST":
        # Preserve all submitted values
        for feature in features:
            input_values[feature] = request.form.get(
                feature, ""
            ).strip()

    else:
        input_values = {
            feature: defaults.get(feature, "")
            for feature in features
        }

    if request.method == "POST":
        try:
            input_data = {}

            for feature in features:
                value = input_values.get(feature, "")

                if value == "":
                    raise ValueError(
                        f"Please enter {feature.replace('_', ' ')}."
                    )

                # Convert numeric inputs to numbers.
                if feature in DROPDOWN_OPTIONS:
                    if value not in DROPDOWN_OPTIONS[feature]:
                        raise ValueError(
                            f"Invalid option for {feature.replace('_', ' ')}."
                        )
                    input_data[feature] = value
                else:
                    number = float(value)

                    if number < 0:
                        raise ValueError(
                            f"{feature.replace('_', ' ')} cannot be negative."
                        )

                    input_data[feature] = number

            # Preserve training feature names and their order
            input_df = pd.DataFrame(
                [input_data],
                columns=features
            )

            # Predict delivery time
            prediction = round(
                float(model.predict(input_df)[0]), 2
            )

            if prediction < 0:
                prediction = 0.0

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        features=features,
        input_values=input_values,
        dropdown_options=DROPDOWN_OPTIONS,
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
