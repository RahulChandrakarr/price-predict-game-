## House Price Prediction (Flask + Linear Regression)

This project is a **simple web app** that predicts a house price from its area using a **linear regression model** built with `scikit-learn`, and a **Flask** UI so you can enter an area in the browser.

When you start the app:
- **The model is trained once at startup** on a tiny example dataset.
- You open a web page where you type an `area`.
- The app returns the **predicted price** and shows the **model MSE** (Mean Squared Error).

---

## How to run

1. **Install dependencies** (from your project folder):

```bash
pip install flask scikit-learn pandas numpy
```

2. **Start the Flask app**:

```bash
python house_price.py
```

3. **Open the browser**:

- Go to `http://127.0.0.1:5000/`
- Enter an area (for example `5`) and click **Predict Price**.

---

## File overview

- **`house_price.py`**: main Python file. Trains the model and defines the Flask routes.
- **`templates/index.html`**: HTML template used to render the form and the prediction result.

---

## `house_price.py` – line‑by‑line explanation

### Imports (lines 1–6)

```python
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

- **`import pandas as pd`**: imports Pandas and gives it the alias `pd` (used to build the dataset as a DataFrame).
- **`import numpy as np`**: imports NumPy and gives it the alias `np` (used to build arrays for prediction).
- **`from flask import Flask, render_template, request`**:
  - **`Flask`**: class used to create the web application object.
  - **`render_template`**: function to render an HTML template (in `templates/index.html`).
  - **`request`**: gives access to incoming HTTP request data (like form inputs).
- **`from sklearn.linear_model import LinearRegression`**: imports the linear regression model class.
- **`from sklearn.model_selection import train_test_split`**: imports utility to split data into train and test sets.
- **`from sklearn.metrics import mean_squared_error`**: imports a function to calculate MSE (a regression error metric).

### Data preparation and model training (lines 9–29)

```python
# Prepare data and train the model once at startup
data = {
    "area": [1, 4, 6, 8, 10],
    "price": [2, 8, 12, 16, 20],
}
```

- **Comment** explains that we will prepare the data and train the model one time when the app starts.
- **`data = { ... }`**: creates a Python dictionary with:
  - Key `"area"` mapped to a list of example area values.
  - Key `"price"` mapped to the corresponding house prices.

```python
df = pd.DataFrame(data)
```

- Converts the `data` dictionary into a **Pandas DataFrame** called `df`.
- This gives us a table with two columns: `area` and `price`.

```python
x = df[["area"]]
y = df["price"]
```

- **`x`**: a DataFrame containing only the `area` column (double brackets keep it as 2D).
- **`y`**: a Series containing the `price` values.
- These will be the **features (`x`)** and **target (`y`)** for the model.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

- Uses **`train_test_split`** to divide the data:
  - **`x_train` / `y_train`**: training data.
  - **`x_test` / `y_test`**: test data.
- **`test_size=0.2`**: 20% of the data goes into the test set (here, 1 sample).
- **`random_state=42`**: sets a fixed random seed so results are reproducible.

```python
model = LinearRegression()
model.fit(x_train, y_train)
```

- **`model = LinearRegression()`**: creates an instance of the linear regression model.
- **`model.fit(x_train, y_train)`**: trains the model using the training data.

```python
predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
```

- **`predictions = model.predict(x_test)`**: uses the trained model to predict prices for the test set `x_test`.
- **`mse = mean_squared_error(y_test, predictions)`**: calculates the mean squared error between the true test prices `y_test` and the predictions.
- **`print(f"Mean Squared Error: {mse}")`**: prints the MSE once when the app starts (useful for debugging/monitoring in the console).

### Flask app setup (lines 32–32)

```python
app = Flask(__name__)
```

- Creates the Flask application object and assigns it to `app`.
- `__name__` tells Flask where to look for resources (like templates) relative to this file.

### Route and view function (lines 35–54)

```python
@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    error = None
```

- **`@app.route("/", methods=["GET", "POST"])`**:
  - Decorator that tells Flask that the function `index` should handle requests to `/` (the home page).
  - Allows both `GET` (initial page load) and `POST` (form submission).
- **`def index():`**: defines the view function for this route.
- Inside the function:
  - **`predicted_price = None`**: initializes the prediction variable (no prediction yet).
  - **`error = None`**: initializes an error message variable (no error yet).

```python
    if request.method == "POST":
        area_value = request.form.get("area", "").strip()
        try:
            area = float(area_value)
            new_area = np.array([[area]])
            predicted_price = model.predict(new_area)[0]
        except ValueError:
            error = "Please enter a valid numeric area."
```

- **`if request.method == "POST":`**:
  - Checks if the request is a form submission.
  - This block only runs when the user submits the form.
- **`area_value = request.form.get("area", "").strip()`**:
  - Reads the value of the input named `"area"` from the form.
  - Defaults to an empty string if not present.
  - `.strip()` removes extra spaces at the beginning/end.
- **`try:`**:
  - Attempts to convert the string to a float and make a prediction.
- **`area = float(area_value)`**:
  - Converts the text input to a floating point number.
- **`new_area = np.array([[area]])`**:
  - Builds a 2D NumPy array of shape `(1, 1)` because `sklearn` expects a 2D array for features.
- **`predicted_price = model.predict(new_area)[0]`**:
  - Calls the trained model to predict the price for the given area.
  - `model.predict(new_area)` returns an array; `[0]` takes the single numeric prediction.
- **`except ValueError:`**:
  - If `float(area_value)` fails (e.g. user typed text, left it empty, etc.), this block runs.
- **`error = "Please enter a valid numeric area."`**:
  - Sets an error message that will be displayed in the template.

```python
    return render_template(
        "index.html",
        predicted_price=predicted_price,
        mse=mse,
        error=error,
    )
```

- **`return render_template(...)`**:
  - Renders the `templates/index.html` file.
  - Passes three variables into the template:
    - **`predicted_price`**: either `None` (no prediction yet) or the numeric prediction.
    - **`mse`**: the Mean Squared Error computed at startup.
    - **`error`**: either `None` or the error message string.
- The template uses these values to show:
  - The prediction (if available).
  - The error message (if any).
  - The model MSE.

### App entry point (lines 57–59)

```python
if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True)
```

- **`if __name__ == "__main__":`**:
  - Ensures the block runs only when this file is executed directly (`python house_price.py`), not when imported.
- **Comment**: explains we are starting the Flask development server.
- **`app.run(debug=True)`**:
  - Starts the Flask server in **debug mode**.
  - Debug mode auto-reloads on code changes and shows detailed error pages during development.

---

## `templates/index.html` – high‑level explanation

- **HTML structure**:
  - A simple, centered card layout with a title “House Price Prediction”.
  - A form with one numeric input:
    - `name="area"` – must match the key used in `request.form.get("area", ...)`.
  - A submit button labeled “Predict Price”.
- **Jinja2 template logic**:
  - Shows an error message if `error` is not `None`.
  - Shows the predicted price (rounded to 2 decimals) if `predicted_price` is not `None`.
  - Shows the MSE (rounded) at the bottom.
- **Styling**:
  - Uses basic CSS for a clean look: centered container, box shadow, spacing, and colors.

#   p r i c e - p r e d i c t - g a m e -  
 