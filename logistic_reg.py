import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Dictionary to store data for each ROI
roi_data = {}


def perform_regression(roi_key, data):
    global roi_data

    # Check if the ROI key already exists in the dictionary
    if roi_key not in roi_data:
        roi_data[roi_key] = []

    # Add the data to the corresponding ROI key in the dictionary
    roi_data[roi_key].extend(data)

    # Check if enough data is available for regression (200 samples)
    if len(roi_data[roi_key]) >= 200:
        # Extract features (X) and target labels (y)
        X = np.array([row[0] for row in roi_data[roi_key]]).reshape(-1, 1)
        y = np.array([row[1] for row in roi_data[roi_key]])

        if np.any(y) and np.any(~y):
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the logistic regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Evaluate the model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            print(f"ROI: {roi_key}")
            print("Train Accuracy:", train_score)
            print("Test Accuracy:", test_score)
