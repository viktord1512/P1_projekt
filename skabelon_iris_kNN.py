from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Indlæs Iris-datasættet
iris = load_iris()
X, y = iris.data, iris.target

# Opdel data i træn- og testdata (80% træning, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialiser og træn Random Forest-modellen
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Forudsig på testdata
y_pred = rf_model.predict(X_test)

# Evaluer modellen
print(f"Nøjagtighed (Accuracy): {accuracy_score(y_test, y_pred):.2f}\n")
print("Klassifikationsrapport:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))