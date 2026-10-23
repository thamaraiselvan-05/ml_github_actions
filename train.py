from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def train_model():
    # Step 1: Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Step 2: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y)

    # Step 3: Create the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42)

    # Step 4: Train the model
    model.fit(X_train, y_train)

    # Step 5: Make predictions
    predictions = model.predict(X_test)

    # Step 6: Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    return model, X_test, y_test, accuracy


if __name__ == "__main__":
    model, X_test, y_test, accuracy = train_model()

    # Save the trained model
    joblib.dump(model, "model.pkl")

    print("Model training completed successfully.")
    print(f"Test accuracy: {accuracy:.2%}")
    print("Model saved as model.pkl")
    print("Thamarai")