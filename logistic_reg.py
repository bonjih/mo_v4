from collections import deque

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def prepare_data(frames_deque):
    # Convert deque to list and then extract features (X) and target values (y)
    X = np.array([frame for frame in frames_deque])  # Features
    y = np.array([frame_label for frame, frame_label in frames_deque])
    print(y)
    return X, y


def perform_regression(frames_deque):
    # Step 1: Prepare data
    X, y = prepare_data(list(frames_deque))

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model
    score = model.score(X_test, y_test)

    return model, score
