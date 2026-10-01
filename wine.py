import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load Iris dataset
data = load_iris()

X = data.data
y = data.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# Start MLflow run
with mlflow.start_run():

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log trained model
    mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    print(f"Model trained with accuracy: {accuracy:.4f}")