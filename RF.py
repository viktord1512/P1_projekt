import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Indlæs heart-datasættet
heart = pd.read_csv("heart.csv")

# Brug kun de første 12 rækker af datasættet for både X og y
X = heart.iloc[:12, :-1].values
y = heart.iloc[:12, -1].values

#onehot encoding for kategoriske features
nonnumeric_features = [1, 2, 8, 10]
numeric_features = [0, 3, 4, 7,]
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(heart.iloc[:, nonnumeric_features]).toarray()

#skaler numeriske features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(heart.iloc[:, numeric_features].values)
X = np.hstack((X_numerical, X_categorical))
y = heart.iloc[:, -1].values

# Opdel data i træn- og testdata (80% træning, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # 0.2 svarer til 20% af dataene
)

# Initialiser og træn Random Forest-modellen
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Forudsig på testdata
y_pred = rf_model.predict(X_test)

# Evaluer modellen
print(f"Nøjagtighed (Accuracy): {accuracy_score(y_test, y_pred):.2f}\n")
print("Klassifikationsrapport:")
print(classification_report(y_test, y_pred))

# confusion matrix med visualization
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)



# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()