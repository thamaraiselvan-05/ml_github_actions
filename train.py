import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,
        test_size=0.2,
        random_state=42,
        stratify=y)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, X_test, y_test, accuracy

if __name__ == "__main__":
    model, X_test, y_test, accuracy = train_model()
    joblib.dump(model, "model.pkl")
    print("Model training completed successfully - CI test!.")
    print(f"Test accuracy: {accuracy:.2%}")
    print("Model saved as model.pkl")