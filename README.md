# Iris Species Classifier using Support Vector Machine

## Overview

This project demonstrates the use of a Support Vector Machine (SVM) to classify iris flowers into their respective species. SVM is a powerful supervised learning algorithm that is well-suited for this task due to the distinct features of the different iris species.

## Features

- Data Loading: Loads the classic Iris dataset, which is a standard for classification tasks.

- Data Splitting: Divides the dataset into training and testing sets for robust model evaluation.

- Hyperparameter Tuning: Utilizes GridSearchCV to automatically find the optimal combination of hyperparameters (like kernel, C, and gamma) to maximize model performance.

- Model Training: Trains an SVM classifier with the best-found parameters.

- Performance Evaluation: Generates a detailed classification report with key metrics like precision, recall, and f1-score.

- Live Prediction: Includes a function to predict the species of a new, unseen data sample.

## Technologies Used

- Python: The core programming language for the project.

- NumPy: Used for efficient numerical operations and array handling.

- scikit-learn: The primary machine learning library for model building, evaluation, and hyperparameter tuning.

## Model Used

The model is a Support Vector Classifier (SVC), which is the scikit-learn implementation of the SVM algorithm for classification. It works by finding the optimal hyperplane that separates data points of different classes.

## Model Training

The dataset is split into an 80% training set and a 20% testing set. A Pipeline is used to chain the model steps, and GridSearchCV performs a comprehensive search over a predefined grid of hyperparameters. This process uses cross-validation on the training data to find the best-performing model configuration before making final predictions on the unseen test data.

## How to Run the Project

1. Clone the repository:

```bash
git clone <https://github.com/sjain2580/Support-Vector-Machine-Classifier>
cd <repository_name>
```

2. Create and activate a virtual environment (optional but recommended):python -m venv venv

- On Windows:
  
```bash
.\venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Script:

```bash
python classifier.py
```

## Contributors

**<https://github.com/sjain2580>**
Feel free to fork this repository, submit issues, or pull requests to improve the project. Suggestions for model enhancement or additional visualizations are welcome!

## Connect with Me

Feel free to reach out if you have any questions or just want to connect!
**[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sjain04/)**
**[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/sjain2580)**
**[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:sjain040395@gmail.com)**

---
