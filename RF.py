from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Indlæs heart-datasættet
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
heart = pd.read_csv("heart.csv")

# Brug kun de første 12 rækker af datasættet for både X og y
X = heart.iloc[:12, :-1].values
y = heart.iloc[:12, -1].values

#onehot encoding for kategoriske features
nonnumeric_features = [1, 2, 6, 8, 10]
numeric_features = [0,3, 4, 5, 7, 9]
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(heart.iloc[:, nonnumeric_features]).toarray()

#skaler numeriske features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(heart.iloc[:, numeric_features].values)
X = np.hstack((X_numerical, X_categorical))
y = heart.iloc[:, -1].values

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
print(classification_report(y_test, y_pred))