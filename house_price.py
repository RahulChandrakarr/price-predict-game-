import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Prepare data and train the model once at startup
data = {
    "area": [1, 4, 6, 8, 10],
    "price": [2, 8, 12, 16, 20],
}

df = pd.DataFrame(data)

x = df[["area"]]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    error = None

    if request.method == "POST":
        area_value = request.form.get("area", "").strip()
        try:
            area = float(area_value)
            new_area = np.array([[area]])
            predicted_price = model.predict(new_area)[0]
        except ValueError:
            error = "Please enter a valid numeric area."

    return render_template(
        "index.html",
        predicted_price=predicted_price,
        mse=mse,
        error=error,
    )


if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True)