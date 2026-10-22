import numpy as np # til numeriske operationer og array-manipulationer
import pandas as pd # til håndtering af datasæt som DataFrame
import seaborn as sns # til udregning af heatmaps, fx confusion matrix
import matplotlib.pyplot as plt # til visualisering af diagrammer, plots og confusion matrixer
from sklearn.ensemble import RandomForestClassifier # til Random Forest-klassifikation
from sklearn.model_selection import train_test_split # til opdeling af datasæt i trænings- og testdata
from sklearn.preprocessing import OneHotEncoder, StandardScaler # til one-hot encoding og standardisering af features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # til evaluering af modelens præstation og beregning af confusion matrix


# Indlæs heart-datasættet
heart = pd.read_csv("heart.csv")

# Onehot encoding for kategoriske features
# nonnumeric_features = [1, 2, 6, 8, 10]
# numeric_features = [0, 3, 4, 5, 7, 9]
nonnumeric_features = [1, 2, 8, 10]
numeric_features = [0, 3, 4, 7,]
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(heart.iloc[:, nonnumeric_features]).toarray()

X_numerical = heart.iloc[:, numeric_features].values

# Lægger de nye one-hot encoded kategoriske features sammen med de numeriske features
X = np.hstack((X_numerical, X_categorical))
y = heart.iloc[:, -1].values

# Opdel data i træn- og testdata (80% træning, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # 0.2 svarer til 20% af dataene som testes
)

# Skaler numeriske features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

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



cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()