# Step 1: Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Step 2: Load the dataset
# Load the Iris dataset from scikit-learn's built-in datasets.
iris = load_iris()

# The 'data' attribute contains the feature matrix (X)
# The 'target' attribute contains the target vector (y)
X = iris.data
y = iris.target

print("Features (X):", X.shape)
print("Target (y):", y.shape)
print("\nFirst 5 samples of features:\n", X[:5])
print("\nFirst 5 samples of target labels:\n", y[:5])
print("\nSpecies names:", iris.target_names)

# Step 3: Split the data into training and testing sets
# We split the data to evaluate the model's performance on unseen data.
# 80% of the data will be used for training, and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# Step 4: Model Improvement with Hyperparameter Tuning
# We will use a Pipeline to combine any potential preprocessing steps with the
# SVM classifier, and then use GridSearchCV to find the best parameters.
pipeline = Pipeline([
    ('classifier', SVC(random_state=42))
])

# Define the grid of hyperparameters to search through.
# We will test two different kernels ('linear' and 'rbf') and a range of 'C' values.
param_grid = {
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto'] # 'gamma' parameter is for 'rbf' kernel
}

# Use GridSearchCV to find the best combination of parameters.
print("\nPerforming GridSearchCV for hyperparameter tuning. This may take a few moments...")
grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=1, cv=5)
grid_search.fit(X_train, y_train)

# The best estimator is the model with the best parameters.
best_model = grid_search.best_estimator_

print("\n--- Hyperparameter Tuning Results ---")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print("Best parameters found:")
print(grid_search.best_params_)
print("-------------------------------------")

# Step 5: Evaluate the best model
# Make predictions on the test set using the best model.
y_pred = best_model.predict(X_test)

# Calculate the accuracy of the model.
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Final Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")

# The classification report provides a more detailed breakdown of performance
# per class, including precision, recall, and f1-score.
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("------------------------")

# Step 6: Make a prediction on new data
# Let's create a hypothetical new sample of an iris flower
# with features: sepal length=5.1, sepal width=3.5, petal length=1.4, petal width=0.2
# This corresponds to an Iris-setosa flower.
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Use the trained model to predict the species of the new sample.
predicted_species_index = best_model.predict(new_sample)
predicted_species_name = iris.target_names[predicted_species_index[0]]

print("\n--- Prediction for a new sample ---")
print(f"Features of the new sample: {new_sample[0]}")
print(f"Predicted species: {predicted_species_name}")
print("-----------------------------------")
